#!/usr/bin/env python3
"""
Profile GNN forward pass performance using torch.profiler.
Identifies bottlenecks and saves detailed profiling results.
"""

import os
import sys
import torch
from torch.profiler import profile, ProfilerActivity, record_function

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phaita.data.icd_conditions import RespiratoryConditions
from phaita.models.gnn_module import SymptomGraphModule


def profile_gnn_forward_pass(batch_size=32, num_warmup=5, num_iterations=20):
    """
    Profile GNN forward pass with torch.profiler.
    
    Args:
        batch_size: Batch size for profiling
        num_warmup: Number of warmup iterations
        num_iterations: Number of profiling iterations
    """
    print(f"🔬 Profiling GNN Forward Pass")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  PyTorch version: {torch.__version__}")
    print()
    
    # Initialize GNN module
    conditions = RespiratoryConditions.get_all_conditions()
    gnn_module = SymptomGraphModule(
        conditions=conditions,
        node_feature_dim=64,
        hidden_dim=128,
        output_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_module = gnn_module.to(device)
    gnn_module.eval()
    
    print(f"✓ GNN module initialized")
    print(f"  Num nodes: {gnn_module.gat.num_nodes}")
    print(f"  Num edges: {gnn_module.edge_index.shape[1]}")
    print()
    
    # Warmup iterations
    print(f"🔥 Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = gnn_module(batch_size=batch_size)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f"✓ Warmup complete")
    print()
    
    # Profile with torch.profiler
    print(f"📊 Profiling ({num_iterations} iterations)...")
    
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    with torch.no_grad():
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("gnn_forward_pass"):
                for _ in range(num_iterations):
                    output = gnn_module(batch_size=batch_size)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
    
    print(f"✓ Profiling complete")
    print()
    
    # Print profiling results
    print("=" * 80)
    print("TOP OPERATIONS BY CPU TIME")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=15
    ))
    print()
    
    if torch.cuda.is_available():
        print("=" * 80)
        print("TOP OPERATIONS BY CUDA TIME")
        print("=" * 80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=15
        ))
        print()
    
    print("=" * 80)
    print("TOP OPERATIONS BY SELF CPU TIME")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=15
    ))
    print()
    
    # Save detailed results to file
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    profile_file = os.path.join(logs_dir, 'gnn_profile.txt')
    
    with open(profile_file, 'w') as f:
        f.write("GNN PROFILING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  PyTorch version: {torch.__version__}\n")
        f.write(f"  Num iterations: {num_iterations}\n")
        f.write(f"  Num nodes: {gnn_module.gat.num_nodes}\n")
        f.write(f"  Num edges: {gnn_module.edge_index.shape[1]}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("TOP OPERATIONS BY CPU TIME\n")
        f.write("=" * 80 + "\n")
        f.write(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=20
        ))
        f.write("\n\n")
        
        if torch.cuda.is_available():
            f.write("TOP OPERATIONS BY CUDA TIME\n")
            f.write("=" * 80 + "\n")
            f.write(prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20
            ))
            f.write("\n\n")
        
        f.write("TOP OPERATIONS BY SELF CPU TIME\n")
        f.write("=" * 80 + "\n")
        f.write(prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=20
        ))
        f.write("\n\n")
        
        f.write("ALL OPERATIONS (sorted by CPU time)\n")
        f.write("=" * 80 + "\n")
        f.write(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=-1
        ))
    
    print(f"✓ Detailed profiling results saved to: {profile_file}")
    print()
    
    # Identify top 3 bottlenecks
    print("=" * 80)
    print("TOP 3 BOTTLENECKS")
    print("=" * 80)
    
    key_averages = prof.key_averages()
    
    # Filter out profiler overhead and get actual operations
    relevant_ops = [
        evt for evt in key_averages 
        if not evt.key.startswith('ProfilerStep') and 
           not evt.key.startswith('enumerate') and
           evt.cpu_time_total > 0
    ]
    
    # Sort by CPU time
    relevant_ops.sort(key=lambda x: x.cpu_time_total, reverse=True)
    
    for i, evt in enumerate(relevant_ops[:3], 1):
        print(f"{i}. {evt.key}")
        print(f"   CPU Time: {evt.cpu_time_total / 1000:.2f} ms")
        print(f"   Self CPU Time: {evt.self_cpu_time_total / 1000:.2f} ms")
        print(f"   Calls: {evt.count}")
        if torch.cuda.is_available() and evt.cuda_time_total > 0:
            print(f"   CUDA Time: {evt.cuda_time_total / 1000:.2f} ms")
        print()
    
    return prof


def benchmark_forward_pass(batch_size=32, num_iterations=100):
    """
    Benchmark forward pass time (ms/batch).
    
    Args:
        batch_size: Batch size for benchmarking
        num_iterations: Number of iterations
        
    Returns:
        Average time per batch in milliseconds
    """
    print(f"⏱️  Benchmarking Forward Pass Time")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")
    print()
    
    # Initialize GNN module
    conditions = RespiratoryConditions.get_all_conditions()
    gnn_module = SymptomGraphModule(
        conditions=conditions,
        node_feature_dim=64,
        hidden_dim=128,
        output_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_module = gnn_module.to(device)
    gnn_module.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = gnn_module(batch_size=batch_size)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = gnn_module(batch_size=batch_size)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / num_iterations
    
    print(f"✓ Benchmark complete")
    print(f"  Total time: {total_time_ms:.2f} ms")
    print(f"  Average time per batch: {avg_time_ms:.4f} ms")
    print(f"  Throughput: {1000 / avg_time_ms:.2f} batches/sec")
    print()
    
    return avg_time_ms


if __name__ == "__main__":
    print("=" * 80)
    print("GNN PROFILING AND BENCHMARKING")
    print("=" * 80)
    print()
    
    # Run profiling
    profile_gnn_forward_pass(batch_size=32, num_warmup=5, num_iterations=20)
    
    print("=" * 80)
    print()
    
    # Run benchmark
    baseline_time = benchmark_forward_pass(batch_size=32, num_iterations=100)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline forward pass time: {baseline_time:.4f} ms/batch")
    print(f"Target after optimization: {baseline_time * 0.7:.4f} ms/batch (30% speedup)")
    print("=" * 80)
