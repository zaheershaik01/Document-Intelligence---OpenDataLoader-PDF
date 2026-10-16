import sys

from negd_docextract.cli import main  

if __name__ == "__main__":  # the guard is REQUIRED on Windows because worker processes re-import this file
    sys.exit(main())
