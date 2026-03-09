#!/usr/bin/env python3
import os

def test_model_port():
    model_port = os.environ.get('MODEL_PORT', 'NOT_SET')
    print(f"MODEL_PORT environment variable: {model_port}")
    print(f"Type: {type(model_port)}")
    
    if model_port != 'NOT_SET':
        try:
            port_int = int(model_port)
            print(f"Port as integer: {port_int}")
        except ValueError:
            print(f"Error: MODEL_PORT '{model_port}' is not a valid integer")
    else:
        print("MODEL_PORT environment variable is not set")

if __name__ == "__main__":
    test_model_port()
