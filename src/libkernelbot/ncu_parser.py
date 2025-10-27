"""
NCU Output Parser
Extracts key metrics from NCU CSV outputs and structures them for LLM analysis
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class NCUParser:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.parsed_data = {
            "metadata": {},
            "summary": {},
            "bottlenecks": [],
            "metrics": {
                "compute": {},
                "memory": {},
                "occupancy": {},
                "control_flow": {}
            },
            "recommendations": []
        }
    
    def parse_csv(self, filename: str) -> List[Dict[str, str]]:
        """Parse a CSV file and return list of dictionaries"""
        filepath = self.output_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found")
            return []
        
        rows = []
        try:
            with open(filepath, 'r') as f:
                # Skip NCU profiling header lines (==PROF==, Max error:, etc.)
                lines = f.readlines()
                csv_start = 0
                for i, line in enumerate(lines):
                    # Find the first line that starts with a quote (CSV header)
                    if line.strip().startswith('"'):
                        csv_start = i
                        break
                
                # Parse from the CSV header onwards
                if csv_start < len(lines):
                    reader = csv.DictReader(lines[csv_start:])
                    for row in reader:
                        rows.append(row)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
        
        return rows
    
    def extract_metric_value(self, rows: List[Dict], metric_name: str) -> Optional[float]:
        """Extract a specific metric value from parsed CSV rows"""
        for row in rows:
            if 'Metric Name' in row and row['Metric Name'] == metric_name:
                try:
                    value_str = row.get('Metric Value', row.get('Avg', ''))
                    # Clean percentage signs and convert
                    value_str = value_str.replace('%', '').replace(',', '').strip()
                    if value_str:
                        return float(value_str)
                except (ValueError, KeyError):
                    continue
        return None
    
    def parse_speed_of_light(self):
        """Parse Speed of Light section for high-level bottleneck identification"""
        rows = self.parse_csv("speed_of_light.csv")
        
        # Key SpeedOfLight metrics - using friendly names from NCU CSV
        sol_metrics = {
            "SM %": "Compute (SM) Throughput",
            "Memory %": "Memory Throughput",
            "DRAM %": "DRAM Throughput",
            "L1/TEX Cache %": "L1/TEX Cache Throughput",
            "L2 Cache %": "L2 Cache Throughput"
        }
        
        for label, metric in sol_metrics.items():
            value = self.extract_metric_value(rows, metric)
            if value is not None:
                self.parsed_data["summary"][label] = value
        
        # Identify primary bottleneck
        if self.parsed_data["summary"]:
            bottleneck = max(self.parsed_data["summary"].items(), key=lambda x: x[1])
            self.parsed_data["bottlenecks"].append({
                "type": bottleneck[0],
                "utilization": bottleneck[1],
                "severity": self._classify_severity(bottleneck[1])
            })
    
    def parse_memory_analysis(self):
        """Parse memory-related metrics"""
        rows = self.parse_csv("memory_analysis.csv")
        
        memory_metrics = {
            "Mem Busy": "Mem Busy",
            "Max Bandwidth": "Max Bandwidth",
            "L1/TEX Hit Rate": "L1/TEX Hit Rate",
            "L2 Hit Rate": "L2 Hit Rate",
            "Mem Pipes Busy": "Mem Pipes Busy"
        }
        
        for label, metric in memory_metrics.items():
            value = self.extract_metric_value(rows, metric)
            if value is not None:
                self.parsed_data["metrics"]["memory"][label] = value
    
    def parse_compute_analysis(self):
        """Parse compute-related metrics"""
        rows = self.parse_csv("compute_analysis.csv")
        
        compute_metrics = {
            "SM Busy": "SM Busy",
            "Issue Slots Busy": "Issue Slots Busy",
            "Executed Ipc Active": "Executed Ipc Active",
            "Executed Ipc Elapsed": "Executed Ipc Elapsed",
            "Issued Ipc Active": "Issued Ipc Active"
        }
        
        for label, metric in compute_metrics.items():
            value = self.extract_metric_value(rows, metric)
            if value is not None:
                self.parsed_data["metrics"]["compute"][label] = value
    
    def parse_occupancy(self):
        """Parse occupancy metrics"""
        rows = self.parse_csv("occupancy.csv")
        
        occupancy_metrics = {
            "Achieved Occupancy": "Achieved Occupancy",
            "Theoretical Occupancy": "Theoretical Occupancy",
            "Achieved Active Warps Per SM": "Achieved Active Warps Per SM",
            "Theoretical Active Warps per SM": "Theoretical Active Warps per SM"
        }
        
        for label, metric in occupancy_metrics.items():
            value = self.extract_metric_value(rows, metric)
            if value is not None:
                self.parsed_data["metrics"]["occupancy"][label] = value
        
        # Check if occupancy is limiting factor
        achieved = self.parsed_data["metrics"]["occupancy"].get("Achieved Occupancy", 0)
        theoretical = self.parsed_data["metrics"]["occupancy"].get("Theoretical Occupancy", 0)
        
        if achieved > 0 and theoretical > 0 and achieved < theoretical * 0.8:
            self.parsed_data["bottlenecks"].append({
                "type": "Low Occupancy",
                "achieved": achieved,
                "theoretical": theoretical,
                "severity": "high" if achieved < theoretical * 0.5 else "medium"
            })
    
    def parse_scheduler_stats(self):
        """Parse warp scheduler statistics (stall reasons)"""
        rows = self.parse_csv("scheduler_stats.csv")
        
        scheduler_metrics = {
            "One or More Eligible": "One or More Eligible",
            "No Eligible": "No Eligible",
            "Active Warps Per Scheduler": "Active Warps Per Scheduler",
            "Eligible Warps Per Scheduler": "Eligible Warps Per Scheduler",
            "Issued Warp Per Scheduler": "Issued Warp Per Scheduler"
        }
        
        for label, metric in scheduler_metrics.items():
            value = self.extract_metric_value(rows, metric)
            if value is not None:
                self.parsed_data["metrics"]["control_flow"][label] = value
        
        # Check for severe scheduler inefficiency
        no_eligible = self.parsed_data["metrics"]["control_flow"].get("No Eligible", 0)
        if no_eligible > 80:  # More than 80% of time with no eligible warps
            self.parsed_data["bottlenecks"].append({
                "type": "Scheduler Stalls",
                "no_eligible_pct": no_eligible,
                "severity": "high" if no_eligible > 90 else "medium"
            })
    
    def parse_warp_stats(self):
        """Parse warp execution efficiency"""
        rows = self.parse_csv("warp_stats.csv")
        
        warp_metrics = {
            "Avg. Active Threads Per Warp": "Avg. Active Threads Per Warp",
            "Avg. Not Predicated Off Threads Per Warp": "Avg. Not Predicated Off Threads Per Warp",
            "Warp Cycles Per Issued Instruction": "Warp Cycles Per Issued Instruction",
            "Warp Cycles Per Executed Instruction": "Warp Cycles Per Executed Instruction"
        }
        
        for label, metric in warp_metrics.items():
            value = self.extract_metric_value(rows, metric)
            if value is not None:
                self.parsed_data["metrics"]["control_flow"][label] = value
        
        # Check for warp divergence (ideal is 32 active threads per warp)
        avg_active = self.parsed_data["metrics"]["control_flow"].get("Avg. Active Threads Per Warp", 32)
        if avg_active < 28:
            self.parsed_data["bottlenecks"].append({
                "type": "Warp Divergence",
                "avg_active_threads": avg_active,
                "efficiency": (avg_active / 32) * 100,
                "severity": "high" if avg_active < 24 else "medium"
            })
    
    def _classify_severity(self, utilization: float) -> str:
        """Classify severity based on utilization percentage"""
        if utilization > 80:
            return "high"
        elif utilization > 50:
            return "medium"
        else:
            return "low"
    
    def generate_recommendations(self):
        """Generate optimization recommendations based on bottlenecks"""
        recommendations = []
        
        for bottleneck in self.parsed_data["bottlenecks"]:
            if "Memory" in bottleneck.get("type", ""):
                recommendations.append({
                    "category": "Memory Optimization",
                    "issue": "High memory throughput utilization",
                    "suggestions": [
                        "Use shared memory to reduce global memory accesses",
                        "Coalesce memory accesses for better bandwidth utilization",
                        "Consider using texture memory for read-only data",
                        "Check for bank conflicts in shared memory"
                    ]
                })
            
            elif "SM" in bottleneck.get("type", ""):
                recommendations.append({
                    "category": "Compute Optimization",
                    "issue": "High SM utilization - compute bound",
                    "suggestions": [
                        "Reduce arithmetic intensity if possible",
                        "Use faster math operations (e.g., __fmul_rn)",
                        "Minimize use of expensive operations (div, sqrt, sin/cos)",
                        "Consider algorithmic optimizations"
                    ]
                })
            
            elif "Occupancy" in bottleneck.get("type", ""):
                recommendations.append({
                    "category": "Occupancy Optimization",
                    "issue": f"Low occupancy (achieved: {bottleneck.get('achieved', 0):.1%})",
                    "suggestions": [
                        "Reduce register usage per thread",
                        "Reduce shared memory usage per block",
                        "Adjust block size to increase occupancy",
                        "Check for launch bound optimization opportunities"
                    ]
                })
            
            elif "Divergence" in bottleneck.get("type", ""):
                recommendations.append({
                    "category": "Control Flow Optimization",
                    "issue": "Warp divergence detected",
                    "suggestions": [
                        "Minimize branching within warps",
                        "Ensure adjacent threads follow same execution path",
                        "Consider data reorganization to reduce divergence",
                        "Use __syncthreads() carefully to avoid serialization"
                    ]
                })
            
            elif "Stall" in bottleneck.get("type", ""):
                stall_type = bottleneck.get("type", "")
                if "Memory Dependency" in stall_type:
                    recommendations.append({
                        "category": "Memory Latency Hiding",
                        "issue": "High memory dependency stalls",
                        "suggestions": [
                            "Increase occupancy to hide memory latency",
                            "Prefetch data to overlap computation with memory access",
                            "Use registers to cache frequently accessed data",
                            "Reorganize computation to reduce memory dependencies"
                        ]
                    })
                elif "Long Scoreboard" in stall_type:
                    recommendations.append({
                        "category": "Instruction Pipeline",
                        "issue": "Long scoreboard stalls (memory or long-latency ops)",
                        "suggestions": [
                            "Increase instruction-level parallelism",
                            "Unroll loops to expose more independent operations",
                            "Interleave memory operations with computation",
                            "Increase occupancy to provide more warps for scheduling"
                        ]
                    })
        
        self.parsed_data["recommendations"] = recommendations
    
    def parse_all(self):
        """Parse all NCU output files"""
        print("[NCU Parser] Parsing NCU outputs...")
        
        self.parse_speed_of_light()
        self.parse_memory_analysis()
        self.parse_compute_analysis()
        self.parse_occupancy()
        self.parse_scheduler_stats()
        self.parse_warp_stats()
        self.generate_recommendations()
        
        print("[NCU Parser] Parsing complete!")
    
    def to_json(self, output_file: Optional[str] = None) -> str:
        """Export parsed data to JSON"""
        json_str = json.dumps(self.parsed_data, indent=2)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_str)
            print(f"[NCU Parser] JSON output saved to: {output_file}")
        
        return json_str
    
    def to_llm_prompt(self, kernel_code: Optional[str] = None) -> str:
        """Generate LLM-ready prompt with profiling data"""
        prompt = "# CUDA Kernel Optimization Analysis\n\n"
        
        if kernel_code:
            prompt += f"## Kernel Code\n```cuda\n{kernel_code}\n```\n\n"
        
        prompt += "## Performance Summary\n"
        for key, value in self.parsed_data["summary"].items():
            prompt += f"- {key}: {value:.1f}%\n"
        
        prompt += "\n## Identified Bottlenecks\n"
        for i, bottleneck in enumerate(self.parsed_data["bottlenecks"], 1):
            prompt += f"{i}. **{bottleneck.get('type', 'Unknown')}** (Severity: {bottleneck.get('severity', 'unknown')})\n"
            for key, value in bottleneck.items():
                if key not in ['type', 'severity']:
                    prompt += f"   - {key}: {value}\n"
        
        prompt += "\n## Detailed Metrics\n"
        for category, metrics in self.parsed_data["metrics"].items():
            if metrics:
                prompt += f"\n### {category.title()}\n"
                for metric, value in metrics.items():
                    prompt += f"- {metric}: {value:.2f}\n"
        
        prompt += "\n## Recommended Optimizations\n"
        for i, rec in enumerate(self.parsed_data["recommendations"], 1):
            prompt += f"\n### {i}. {rec['category']}\n"
            prompt += f"**Issue**: {rec['issue']}\n\n"
            prompt += "**Suggestions**:\n"
            for suggestion in rec['suggestions']:
                prompt += f"- {suggestion}\n"
        
        prompt += "\n---\n"
        prompt += "Based on the above profiling data, provide 3-5 specific, actionable optimization tips "
        prompt += "for this kernel. Focus on the most impactful changes first.\n"
        
        return prompt

