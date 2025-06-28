#!/bin/bash

# STAMP Benchmark - Run All Frameworks Script
# This script runs benchmarks across PyTorch, TensorFlow, and JAX

set -e

echo "🚀 STAMP Benchmark - Running All Frameworks"
echo "=============================================="

# Default parameters
SIZES="512,1024,2048"
PRECISIONS="fp32,mixed"
OUTPUT_DIR="logs"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sizes)
            SIZES="$2"
            shift 2
            ;;
        --precisions)
            PRECISIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sizes SIZES             Comma-separated matrix sizes (default: 512,1024,2048)"
            echo "  --precisions PRECISIONS   Comma-separated precision modes (default: fp32,mixed)"
            echo "  --output-dir DIR          Output directory for logs (default: logs)"
            echo "  --verbose                 Enable verbose output"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with default settings"
            echo "  $0 --sizes 1024,2048 --verbose       # Custom sizes with verbose output"
            echo "  $0 --precisions fp16,fp32,mixed      # Test multiple precisions"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Convert comma-separated values to arrays
IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"
IFS=',' read -ra PRECISION_ARRAY <<< "$PRECISIONS"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set verbose flag for Python script
VERBOSE_FLAG=""
if [ "$VERBOSE" = true ]; then
    VERBOSE_FLAG="--verbose"
fi

echo "Configuration:"
echo "  Sizes: ${SIZE_ARRAY[*]}"
echo "  Precisions: ${PRECISION_ARRAY[*]}"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Verbose: $VERBOSE"
echo ""

# Function to run benchmark for a specific framework
run_framework_benchmark() {
    local framework=$1
    echo "📊 Running $framework benchmarks..."
    
    for size in "${SIZE_ARRAY[@]}"; do
        for precision in "${PRECISION_ARRAY[@]}"; do
            echo "  🔄 Testing $framework: size=$size, precision=$precision"
            
            # Run the benchmark
            python main.py \
                --framework "$framework" \
                --precision "$precision" \
                --output-dir "$OUTPUT_DIR" \
                --size "$size" \
                $VERBOSE_FLAG
            
            if [ $? -eq 0 ]; then
                echo "  ✅ $framework benchmark completed successfully"
            else
                echo "  ❌ $framework benchmark failed"
                # Continue with other benchmarks even if one fails
            fi
        done
    done
    echo ""
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not available. Please install Python first."
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Please run this script from the STAMP benchmark root directory."
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import torch, tensorflow, jax" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some ML frameworks are not installed. Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please install manually."
        exit 1
    fi
fi

# Start benchmarking
echo "🏁 Starting benchmark execution..."
START_TIME=$(date +%s)

# Run benchmarks for each framework
run_framework_benchmark "pytorch"
run_framework_benchmark "tensorflow" 
run_framework_benchmark "jax"

# Calculate total runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo "🎉 All benchmarks completed!"
echo "📈 Total runtime: ${RUNTIME} seconds"
echo "📁 Results saved to: $OUTPUT_DIR/"

# Generate summary report
echo "📋 Generating summary report..."
python -c "
import os
import json
from pathlib import Path
from datetime import datetime

log_dir = Path('$OUTPUT_DIR')
log_files = list(log_dir.glob('*.json'))

if log_files:
    print(f'Found {len(log_files)} result files:')
    for log_file in sorted(log_files):
        print(f'  - {log_file.name}')
    
    # Try to create a simple summary
    try:
        latest_file = max(log_files, key=os.path.getctime)
        with open(latest_file) as f:
            data = json.load(f)
        
        if 'results' in data:
            results = data['results']
            frameworks = set(r.get('framework', 'unknown') for r in results)
            operations = set(r.get('operation', 'unknown') for r in results)
            
            print(f'\\n📊 Summary of latest results ({latest_file.name}):')
            print(f'  Frameworks tested: {', '.join(sorted(frameworks))}')
            print(f'  Operations: {', '.join(sorted(operations))}')
            print(f'  Total benchmarks: {len(results)}')
            
            # Show performance highlights
            if results:
                fastest = min(results, key=lambda x: x.get('mean_time', float('inf')))
                slowest = max(results, key=lambda x: x.get('mean_time', 0))
                
                print(f'\\n⚡ Fastest: {fastest.get('framework', 'N/A')} {fastest.get('operation', 'N/A')} ({fastest.get('mean_time', 0):.4f}s)')
                print(f'🐌 Slowest: {slowest.get('framework', 'N/A')} {slowest.get('operation', 'N/A')} ({slowest.get('mean_time', 0):.4f}s)')
    except Exception as e:
        print(f'Could not generate detailed summary: {e}')
else:
    print('No result files found.')
"

echo ""
echo "🎯 Next steps:"
echo "  1. Review results in the $OUTPUT_DIR/ directory"
echo "  2. Use 'python scripts/analyze_results.py' for detailed analysis"
echo "  3. Start the dashboard with 'streamlit run scripts/dashboard.py'"
echo ""
echo "Thank you for using STAMP! 🙏"
