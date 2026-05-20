from datetime import datetime

def structure_output(source_type, raw_input, processed_output):
    return {
        "timestamp": datetime.now().isoformat(),
        "source": source_type,  # "EEG" or "Speech"
        "input_raw": raw_input.tolist() if hasattr(raw_input, "tolist") else raw_input,
        "output_text": processed_output
    }
