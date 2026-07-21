import os

def find_files():
    print("Searching for database files on Z:...")
    for root, dirs, files in os.walk("Z:\\"):
        for file in files:
            if file.endswith((".db", ".sqlite", ".sqlite3")):
                full_path = os.path.join(root, file)
                print(f"Found: {full_path} ({os.path.getsize(full_path)} bytes)")

if __name__ == "__main__":
    find_files()
