import argparse
import sys
import time
from pathlib import Path

def run_phase1():                       # Run Data Exploration script
    
    print(" Running Phase 1: Data Exploration & Quality Assessment")
    
    try:
        # Import and run phase 1
        sys.path.append(str(Path(__file__).parent.parent))
        from src.data_exploration import DataExplorer
        
        analyzer = DataExplorer(
            data_path="data/raw/Telco-Customer-data.csv",
            output_dir="reports/"
        )
        analyzer.run_complete_analysis()
        print(" Phase 1 completed successfully!")
        return True
        
    except Exception as e:
        print(f" Phase 1 failed: {str(e)}")
        return False

def run_phase2():
    """Run Phase 2: Feature Engineering"""
    print(" Running Phase 2: Feature Engineering")
    print(" Phase 2 implementation coming in next iteration")
    return True

def run_phase3():
    """Run Phase 3: Model Development"""
    print(" Running Phase 3: Model Development")
    print(" Phase 3 implementation coming in next iteration")
    return True

def run_all_phases():
    """Run all phases sequentially"""
    print(" Running All Phases of TeleRetain")
    print("=" * 50)
    
    phases = [
        ("Phase 1", run_phase1),
        ("Phase 2", run_phase2), 
        ("Phase 3", run_phase3)
    ]
    
    results = []
    total_start_time = time.time()
    
    for phase_name, phase_func in phases:
        print(f"\n{'='*20} {phase_name} {'='*20}")
        start_time = time.time()
        
        success = phase_func()
        
        end_time = time.time()
        duration = end_time - start_time
        
        results.append({
            'phase': phase_name,
            'success': success,
            'duration': duration
        })
        
        status = " SUCCESS" if success else "❌ FAILED"
        print(f"{phase_name} completed in {duration:.2f}s - {status}")
        
        if not success:
            print(f"❌ Stopping execution due to {phase_name} failure")
            break
    
    # Summary
    total_duration = time.time() - total_start_time         
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    print(f"Total execution time: {total_duration:.2f}s")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['phase']}: {result['duration']:.2f}s")
    
    successful_phases = sum(1 for r in results if r['success'])
    print(f"\nCompleted: {successful_phases}/{len(results)} phases")

def main():
    parser = argparse.ArgumentParser(description="Run TeleRetain experiments")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], 
                       help="Run specific phase")
    parser.add_argument("--all-phases", action="store_true",
                       help="Run all phases sequentially")
    
    args = parser.parse_args()
    
    if args.all_phases:
        run_all_phases()
    elif args.phase == 1:
        run_phase1()
    elif args.phase == 2:
        run_phase2()
    elif args.phase == 3:
        run_phase3()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()