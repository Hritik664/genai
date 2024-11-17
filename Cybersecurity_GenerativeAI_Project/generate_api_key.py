import secrets

def generate_api_key():
    return secrets.token_hex(32)

if __name__ == "__main__":
    api_key = generate_api_key()
    print(f"Your generated API key: {api_key}")
