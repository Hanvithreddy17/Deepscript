import urllib.request
import urllib.error
import json
import os

def post_file(url, filepath):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(filepath)}"\r\n'.encode("utf-8"))
    body.extend(f"Content-Type: image/jpeg\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    
    req = urllib.request.Request(url, data=bytes(body), headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")

def main():
    print("Testing live server at http://localhost:8000/predict ...")
    tests = [
        "dataset/whole_inscriptions/tamil_brahmi/TBSI_020_original.jpg",
        "dataset/whole_inscriptions/brahmi/Devanampiyena Piyadasi in the Lumbini Edict of Ashoka.jpg",
        "dataset/negative_evaluation_set/ordinary_objects/spidey.jpg",
        "dataset/negative_evaluation_set/blank_and_low_info/pure_black.png",
    ]

    for t in tests:
        code, res = post_file("http://localhost:8000/predict?top_k=5", t)
        if isinstance(res, dict):
            print(f"[{os.path.basename(t)}]: HTTP {code} | status={res.get('status')} | decision={res.get('decision')} | script={res.get('script')} | confidence={res.get('confidence')}")
        else:
            print(f"[{os.path.basename(t)}]: HTTP {code} | {res[:80]}")

    # Test Empty File
    empty_req = urllib.request.Request(
        "http://localhost:8000/predict?top_k=5",
        data=b"",
        headers={"Content-Type": "multipart/form-data; boundary=----WebKitFormBoundaryEmpty"},
    )
    try:
        urllib.request.urlopen(empty_req)
    except urllib.error.HTTPError as e:
        print(f"[Empty file (0 bytes)]: HTTP {e.code} | {e.read().decode('utf-8')[:80]}")

if __name__ == "__main__":
    main()
