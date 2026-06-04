from PIL import Image

def run_vlm_inference(image: Image.Image, version: str, model, processor, prompt: str = "") -> str:
    if version not in ["Moondream2 (2B)", "Moondream2 (0.5B)"]:
        raise ValueError(f"Unsupported model version: {version}")
        
    if not prompt or not prompt.strip():
        prompt = "Describe what you see in this image briefly and clearly. Focus on the main subject, people, objects, and any important context. Keep it under 3 sentences."
        
    # Check for the newer API (model.query) or fallback to older API (encode_image / answer_question)
    if hasattr(model, "query"):
        response = model.query(image, prompt)
        return response["answer"]
    else:
        # Fallback to older Moondream version API
        enc_image = model.encode_image(image)
        return model.answer_question(enc_image, prompt, processor)
