# file: text_infer_server.py
# 简单 TCP 服务：
#   Unity 发送一行文本（描述） -> 服务器返回 "label|prob\n"

import socketserver
from tools.run_model import load_model, infer_once
HOST = "127.0.0.1"
PORT = 9009

print("Loading text model...")
clf, label_encoder, vectorizer = load_model()
print("Model loaded. Listening on %s:%d" % (HOST, PORT))


class TextInferHandler(socketserver.StreamRequestHandler):
    def handle(self):
        client_ip, client_port = self.client_address
        print(f"[+] Connection from {client_ip}:{client_port}")

        while True:
            line = self.rfile.readline()
            if not line:
                print(f"[-] {client_ip}:{client_port} disconnected")
                break

            query = line.decode("utf-8").strip()
            if not query:
                continue

            print(f"Query from {client_ip}:{client_port}: {query!r}")

            try:
                results = infer_once(query, clf, label_encoder, vectorizer, top_k=1)
                top = results[0]
                label = top["label"]
                prob = top["prob"]

                resp = f"{label}"
                print(f"[SERVER] Query: {query!r}")
                print(f"[SERVER] Response: {resp!r}", flush=True)

            except Exception as e:
                print("Error during inference:", e)
                resp = "ERROR|0.0\n"

            self.wfile.write(resp.encode("utf-8"))


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    with ThreadedTCPServer((HOST, PORT), TextInferHandler) as server:
        print("Server started.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
