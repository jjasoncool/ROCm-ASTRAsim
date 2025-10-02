#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified ROCm Compatibility Module

Provides unified interface for GPU frequency monitoring across ROCm versions.
Supports both amd-smi (ROCm 7.0+) and rocm-smi (ROCm 6.x) with automatic detection.
"""

import subprocess
import json
import re
import logging
from typing import Optional, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ROCmTool(Enum):
    AMD_SMI = "amd-smi"      # ROCm 7.0+
    ROCM_SMI = "rocm-smi"    # ROCm 6.x
    NONE = "none"

@dataclass
class GPUFrequencyInfo:
    """GPU frequency information."""
    device_id: int
    sclk_mhz: Optional[float] = None   # Shader/Graphics clock
    mclk_mhz: Optional[float] = None   # Memory clock
    fclk_mhz: Optional[float] = None   # Fabric clock
    socclk_mhz: Optional[float] = None # SOC clock
    dcefclk_mhz: Optional[float] = None # DCE clock
    tool_used: str = "unknown"
    timestamp: Optional[float] = None  # For compatibility

class ROCmCompat:
    """Simplified ROCm compatibility layer."""

    def __init__(self):
        self.tool = self._detect_tool()
        self.version = "unknown"
        if self.tool != ROCmTool.NONE:
            try:
                self.version = self._get_version()
            except:
                pass

    def _detect_tool(self) -> ROCmTool:
        """Detect available ROCm tool."""
        # Try amd-smi first (newer)
        try:
            subprocess.run(["amd-smi", "version"],
                         check=True, capture_output=True, timeout=5)
            return ROCmTool.AMD_SMI
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback to rocm-smi
        try:
            subprocess.run(["rocm-smi", "--version"],
                         check=True, capture_output=True, timeout=5)
            return ROCmTool.ROCM_SMI
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return ROCmTool.NONE

    def _get_version(self) -> str:
        """Get tool version."""
        try:
            if self.tool == ROCmTool.AMD_SMI:
                result = subprocess.run(["amd-smi", "version"],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    output = result.stdout.strip()
                    # Parse: "AMDSMI Tool: 25.5.1+41065ee6 | AMDSMI Library version: 25.5.1 | ROCm version: 6.4.3"
                    import re
                    # Extract AMDSMI Tool version
                    match = re.search(r'AMDSMI Tool:\s*([^\s|]+)', output)
                    if match:
                        return f"amd-smi {match.group(1)}"
                    # Fallback to library version
                    match = re.search(r'AMDSMI Library version:\s*([^\s|]+)', output)
                    if match:
                        return f"amd-smi {match.group(1)}"
                    # Fallback to ROCm version
                    match = re.search(r'ROCm version:\s*([^\s|]+)', output)
                    if match:
                        return f"ROCm {match.group(1)}"

            elif self.tool == ROCmTool.ROCM_SMI:
                result = subprocess.run(["rocm-smi", "--version"],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    output = result.stdout.strip()
                    # Parse rocm-smi version output
                    import re
                    match = re.search(r'version:\s*([^\s]+)', output)
                    if match:
                        return f"rocm-smi {match.group(1)}"

        except Exception as e:
            logger.debug(f"Failed to get version: {e}")
            pass
        return "unknown"

    def is_available(self) -> bool:
        """Check if any ROCm tool is available."""
        return self.tool != ROCmTool.NONE

    def get_tool_info(self) -> Dict[str, str]:
        """Get tool information."""
        return {
            "tool": self.tool.value,
            "version": self.version,
            "available": self.is_available()
        }

    def get_version_info(self):
        """Legacy compatibility method."""
        class VersionInfo:
            def __init__(self, tool_type, tool_version):
                self.tool_type = tool_type
                self.tool_version = tool_version

        return VersionInfo(self.tool, self.version)

    def get_gpu_frequency(self, device_id: int = 0) -> Optional[GPUFrequencyInfo]:
        """Get GPU frequency for specified device."""
        if not self.is_available():
            return None

        if self.tool == ROCmTool.AMD_SMI:
            return self._query_amd_smi(device_id)
        elif self.tool == ROCmTool.ROCM_SMI:
            return self._query_rocm_smi(device_id)

        return None

    def _query_amd_smi(self, device_id: int) -> Optional[GPUFrequencyInfo]:
        """Query GPU frequency using amd-smi."""
        try:
            # Use text output which is more reliable
            result = subprocess.run([
                "amd-smi", "metric", "-g", str(device_id), "-c"
            ], check=True, capture_output=True, text=True, timeout=10)

            return self._parse_amd_smi_text(result.stdout, device_id)

        except Exception as e:
            logger.debug(f"amd-smi query failed: {e}")
            return None

    def _extract_frequency_from_section(self, lines: list, start_idx: int, fallback_to_min: bool = False) -> Optional[float]:
        """通用頻率解析函數"""
        for j in range(start_idx + 1, min(start_idx + 10, len(lines))):
            line = lines[j].strip()
            if line.startswith("CLK:"):
                match = re.search(r'CLK:\s*(\d+)\s*MHz', line)
                if match:
                    freq = float(match.group(1))
                    if freq > 0:
                        return freq
                    elif fallback_to_min:
                        # 如果是0，嘗試找MIN_CLK
                        for k in range(j + 1, min(j + 5, len(lines))):
                            min_line = lines[k].strip()
                            if min_line.startswith("MIN_CLK:"):
                                min_match = re.search(r'MIN_CLK:\s*(\d+)\s*MHz', min_line)
                                if min_match:
                                    return float(min_match.group(1))
                    return freq if freq > 0 else None
                break
        return None

    def _parse_amd_smi_text(self, output: str, device_id: int) -> Optional[GPUFrequencyInfo]:
        """Parse amd-smi text output for frequency."""
        try:
            lines = output.split('\n')
            freqs = {}

            # 定義要解析的section和對應的頻率名稱
            sections = {
                "GFX_0:": "sclk_mhz",
                "MEM_0:": "mclk_mhz",
                "FCLK_0:": "fclk_mhz",
                "SOCCLK_0:": "socclk_mhz",
                "DCLK_0:": "dcefclk_mhz"  # 用DCLK作為DCE的近似值
            }

            for i, line in enumerate(lines):
                line = line.strip()
                if line in sections:
                    freq_name = sections[line]
                    # GFX_0 如果是0要fallback到MIN_CLK
                    fallback = (freq_name == "sclk_mhz")
                    freq = self._extract_frequency_from_section(lines, i, fallback)
                    if freq is not None:
                        freqs[freq_name] = freq

            # 至少要有SCLK才返回結果
            if freqs.get("sclk_mhz", 0) > 0:
                import time
                return GPUFrequencyInfo(
                    device_id=device_id,
                    sclk_mhz=freqs.get("sclk_mhz"),
                    mclk_mhz=freqs.get("mclk_mhz"),
                    fclk_mhz=freqs.get("fclk_mhz"),
                    socclk_mhz=freqs.get("socclk_mhz"),
                    dcefclk_mhz=freqs.get("dcefclk_mhz"),
                    tool_used="amd-smi",
                    timestamp=time.time()
                )

            return None

        except Exception as e:
            logger.debug(f"Failed to parse amd-smi output: {e}")
            return None

    def _query_rocm_smi(self, device_id: int) -> Optional[GPUFrequencyInfo]:
        """Query GPU frequency using rocm-smi."""
        try:
            result = subprocess.run([
                "rocm-smi", "--showclocks", "--json"
            ], check=True, capture_output=True, text=True, timeout=10)

            data = json.loads(result.stdout)
            return self._parse_rocm_smi_json(data, device_id)

        except Exception as e:
            logger.debug(f"rocm-smi query failed: {e}")
            return None

    def _parse_rocm_smi_json(self, data: Dict, device_id: int) -> Optional[GPUFrequencyInfo]:
        """Parse rocm-smi JSON output for frequency."""
        try:
            gpu_data = next(iter(data.values()))

            # 定義要解析的鍵值和對應的頻率名稱
            freq_mapping = {
                "sclk_mhz": ["sclk clock speed:", "sclk", "gfx_clock"],
                "mclk_mhz": ["mclk clock speed:", "mclk", "mem_clock"],
                "fclk_mhz": ["fclk clock speed:", "fclk"],
                "socclk_mhz": ["socclk clock speed:", "socclk"],
                "dcefclk_mhz": ["dcefclk clock speed:", "dcefclk"]
            }

            freqs = {}
            for freq_name, possible_keys in freq_mapping.items():
                for key in possible_keys:
                    if key in gpu_data:
                        freq = self._parse_clock_value(gpu_data[key])
                        if freq and freq > 0:
                            freqs[freq_name] = freq
                            break

            # 如果SCLK是0，設為最小值避免None
            if freqs.get("sclk_mhz", 0) == 0:
                freqs["sclk_mhz"] = 100.0  # 最小值表示GPU空閒但可用

            if freqs.get("sclk_mhz", 0) > 0:
                import time
                return GPUFrequencyInfo(
                    device_id=device_id,
                    sclk_mhz=freqs.get("sclk_mhz"),
                    mclk_mhz=freqs.get("mclk_mhz"),
                    fclk_mhz=freqs.get("fclk_mhz"),
                    socclk_mhz=freqs.get("socclk_mhz"),
                    dcefclk_mhz=freqs.get("dcefclk_mhz"),
                    tool_used="rocm-smi",
                    timestamp=time.time()
                )

            return None

        except Exception as e:
            logger.debug(f"Failed to parse rocm-smi output: {e}")
            return None

    def _parse_clock_value(self, value) -> Optional[float]:
        """Parse clock value from various formats."""
        if isinstance(value, (int, float)):
            return float(value) if value > 0 else None

        if isinstance(value, str):
            # Extract numeric value from string like "1500MHz" or "(1500Mhz)"
            match = re.search(r'(\d+(?:\.\d+)?)', value.replace('(', '').replace(')', ''))
            if match:
                return float(match.group(1))

        if isinstance(value, dict):
            # Try common keys in nested dict
            for key in ['current', 'avg', 'max', 'value']:
                if key in value:
                    return self._parse_clock_value(value[key])

        return None

# Convenience functions
def get_gpu_frequency(device_id: int = 0) -> Optional[GPUFrequencyInfo]:
    """Get GPU frequency using auto-detected ROCm tool."""
    rocm = ROCmCompat()
    return rocm.get_gpu_frequency(device_id)

def is_rocm_available() -> bool:
    """Check if any ROCm monitoring tool is available."""
    rocm = ROCmCompat()
    return rocm.is_available()
