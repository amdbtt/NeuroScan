#!/bin/zsh
cd "/Users/ahmad/Desktop/Desktop/Semester_Projects/ML Project" || exit 1
echo "Starting NeuroScan on http://127.0.0.1:5001 ..."
.venv/bin/python3 -u -c "
from app import app
print('Open http://127.0.0.1:5001 in your browser', flush=True)
app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
"
