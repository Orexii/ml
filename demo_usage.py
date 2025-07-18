#!/usr/bin/env python3
"""
Trading Strategy Simulator - Usage Examples
============================================

This script demonstrates different ways to use the enhanced trading simulator.
"""

import subprocess
import sys
import time

def run_command(description, command):
    """Run a command and display the description"""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"{'='*60}")
    print(f"💻 Command: {command}")
    print(f"⏱️  Running...")
    
    try:
        result = subprocess.run(command, shell=True, cwd='/home/malmorga/ml', 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Success!")
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                print("📄 Output (last 10 lines):")
                for line in lines[-10:]:
                    print(f"   {line}")
            else:
                print("📄 Output:")
                print(result.stdout)
        else:
            print("❌ Error!")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 2 minutes")
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    """Demonstrate different usage patterns"""
    
    print("Trading Strategy Simulator - Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "description": "1. Quick Test (2 specific symbols, limited data)",
            "command": "/home/malmorga/ml/.venv/bin/python trading_simulator.py --preset quick_test"
        },
        {
            "description": "2. Custom Symbol Selection",
            "command": "/home/malmorga/ml/.venv/bin/python trading_simulator.py --symbols AAPL,MSFT --max-rows 5000"
        },
        {
            "description": "3. Test First 3 Symbols with Compact Output",
            "command": "/home/malmorga/ml/.venv/bin/python trading_simulator.py --count 3 --max-rows 3000"
        },
        {
            "description": "4. Sample Run with Tech Stocks (if available)",
            "command": "/home/malmorga/ml/.venv/bin/python trading_simulator.py --symbols AAPL,MSFT,AMZN --max-rows 5000 --no-charts"
        }
    ]
    
    for example in examples:
        run_command(example["description"], example["command"])
        print("\n" + "🔄 " + "Moving to next example..." + "\n")
        time.sleep(2)  # Brief pause between examples
    
    print(f"\n{'='*60}")
    print("🎉 ALL EXAMPLES COMPLETED!")
    print(f"{'='*60}")
    print("📊 Key Features Demonstrated:")
    print("   ✅ Preset configurations")
    print("   ✅ Custom symbol selection") 
    print("   ✅ Flexible row limits")
    print("   ✅ Verbose and compact output modes")
    print("   ✅ Chart generation control")
    print("")
    print("🔧 Available Command Line Options:")
    print("   --preset {quick_test, major_stocks, tech_stocks, sample_run}")
    print("   --symbols SYMBOL1,SYMBOL2,...")
    print("   --max-rows NUMBER")
    print("   --count NUMBER")
    print("   --verbose")
    print("   --no-charts")
    print("   --exclude SYMBOL1,SYMBOL2,...")

if __name__ == "__main__":
    main()
