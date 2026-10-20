#!/usr/bin/env bash
# One-time setup. Run from the repository root (the folder with src):
#   bash scripts/setup.sh
set -e

# Refuse to run from the wrong folder, with a hint. A common cause is
# extracting the zip one level too deep (project/ project/).
if [ ! -f requirements.txt ]; then
    echo "ОШИБКА: ты не в той папке — здесь нет requirements.txt."
    echo "Текущая папка: $(pwd)"
    for d in */; do
        if [ -f "${d}requirements.txt" ]; then
            echo "Похоже, проект внутри: ${d}"
            echo "Выполни: cd ${d} && bash scripts/setup.sh"
            exit 1
        fi
    done
    echo "Найди папку, где лежит src, и запусти скрипт оттуда."
    exit 1
fi

echo "== Python =="
python3 --version

echo "== Virtual environment =="
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "created .venv"
fi
source .venv/bin/activate

echo "== Dependencies =="
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== Database =="
python src/db.py

echo
echo "Готово. Дальше:"
echo "  source .venv/bin/activate"
echo "  python src/collect.py"
echo "  python src/youtube_collect.py --seed"
echo "  python src/origins.py"
echo "  python src/report.py"
echo "  streamlit run dashboard/app.py"
