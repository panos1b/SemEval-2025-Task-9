You must install DeepSeek Locally to run this script

conda activate python3
pip install ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:8b
ollama run deepseek-r1:8b

curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:8b",
  "messages": [{ "role": "user", "content": "Solve: 25 * 25" }],
  "stream": false
}'