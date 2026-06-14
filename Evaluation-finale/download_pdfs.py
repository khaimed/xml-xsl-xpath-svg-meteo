"""Telecharge les 4 PDF de la base documentaire (education financiere
personnelle) dans data/pdfs/.

Lancer avec : uv run python download_pdfs.py
"""
import urllib.request
from pathlib import Path

PDF_DIR = Path(__file__).resolve().parent / "data" / "pdfs"

SOURCES = {
    "guide_pedagogique_finance_pour_tous.pdf": (
        "https://www.lafinancepourtous.com/IMG/pdf/guide_pedagogique/guide-pedagogique.pdf"
    ),
    "quest_ce_que_education_financiere.pdf": (
        "https://www.lafinancepourtous.com/IMG/pdf/wikipedia.pdf"
    ),
    "amf_investir_votre_epargne.pdf": (
        "https://www.amf-france.org/sites/institutionnel/files/resource/"
        "Lire%20le%20guide%20pedagogique%20%20Investir%20votre%20epargne%20etape%20par%20etape%20.pdf"
    ),
    "guide_surendettement.pdf": (
        "https://www.iedom.fr/IMG/pdf/guide_du_surendettement.pdf"
    ),
}


def download_all() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        dest = PDF_DIR / filename
        if dest.exists():
            print(f"OK (deja present) : {filename}")
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            dest.write_bytes(response.read())
        size_kb = dest.stat().st_size / 1024
        print(f"Telecharge : {filename} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    download_all()
