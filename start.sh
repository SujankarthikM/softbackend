#!/bin/bash

# Start the first Flask app on port 5000
gunicorn -b 0.0.0.0:5000 app:app &

# Start the second Flask app on port 5001
gunicorn -b 0.0.0.0:5001 j:app &

# Start the third Flask app on port 5002
gunicorn -b 0.0.0.0:5002 app1:app &

# Wait for all background processes to finish
wait
