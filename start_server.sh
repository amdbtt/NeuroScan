#!/bin/zsh
cd "/Users/ahmad/Desktop/Desktop/Semester_Projects/ML Project" || exit 1
exec .venv/bin/python3 -u -c "
from app import app
print('NeuroScan ready at http://127.0.0.1:5001', flush=True)
app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
"
