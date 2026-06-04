from src.app import demo, registry

if __name__ == "__main__":
    # Pre-cache registry setup on startup with default Moondream2 (2B)
    print("Warm-starting ModelRegistry with Moondream2 (2B)...")
    registry.get_vlm("Moondream2 (2B)")
    
    demo.queue().launch()
