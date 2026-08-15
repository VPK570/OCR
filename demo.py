"""
Quick Demo Script for RxScanner

Run this to quickly test the prescription OCR pipeline.
Usage:
    python demo.py                      # Demo mode with sample data
    python demo.py --image path/to/rx   # Process specific image
    python demo.py --frontend           # Start Streamlit frontend
"""

import argparse
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))


def run_demo():
    """Run a quick demo with sample data."""
    print("=" * 60)
    print("RxScanner Demo - Prescription OCR")
    print("=" * 60)

    # Import the abbreviation module
    from pipeline.medical.abbreviations import expand_text, get_abbreviation_count

    # Sample prescription text
    sample_texts = [
        "Metformin 500mg BD",
        "Take TDS after meals PO",
        "Paracetamol 650mg QID PRN for pain",
        "Omeprazole 20mg OD before breakfast",
    ]

    print("\n📝 Sample Prescriptions:")
    print("-" * 60)

    for text in sample_texts:
        expanded = expand_text(text)
        abbrev_count = get_abbreviation_count(text)

        print(f"\n📄 Original: {text}")
        print(f"   Expanded: {expanded}")
        print(f"   Abbreviations found: {abbrev_count}")

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


def run_frontend():
    """Start the Streamlit frontend."""
    print("Starting RxScanner Frontend...")
    print("Open http://localhost:8501 in your browser")
    os.system("streamlit run frontend/app.py --server.headless true")


def process_image(image_path: str):
    """Process a specific prescription image."""
    print(f"Processing: {image_path}")
    # TODO: Implement actual pipeline call
    print("Pipeline integration coming soon!")


def main():
    parser = argparse.ArgumentParser(description="RxScanner Demo")
    parser.add_argument("--image", "-i", help="Path to prescription image")
    parser.add_argument("--frontend", "-f", action="store_true", help="Start frontend")
    parser.add_argument("--demo", "-d", action="store_true", help="Run demo mode")

    args = parser.parse_args()

    if args.frontend:
        run_frontend()
    elif args.image:
        process_image(args.image)
    else:
        # Default: run demo
        run_demo()
        print("\n💡 Tip: Use --frontend to start the web interface")
        print("   Use --image <path> to process a specific image")


if __name__ == "__main__":
    main()