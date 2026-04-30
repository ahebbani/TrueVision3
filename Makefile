.PHONY: run run-audio run-face run-force-both run-server run-server-dev db setup-pi setup-server

run:
	python3 main.py

run-audio:
	python3 main.py --force-mode audio

run-face:
	python3 main.py --force-mode face

run-force-both:
	python3 main.py --force-mode both

run-server:
	python3 -m server.app

run-server-dev:
	uvicorn server.app:app --reload --host 0.0.0.0 --port 8008

db:
	python3 scripts/visualize_db.py

setup-pi:
	bash setup_pi.sh

setup-server:
	bash setup_server.sh
