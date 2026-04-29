# Pre-registration PDF generator
# Requirements (Ubuntu/Debian):
#   apt-get install -y pandoc wkhtmltopdf fonts-noto-cjk
#
# Usage:
#   make           # build both PDFs
#   make jp        # build Japanese master PDF only
#   make en        # build English version PDF only
#   make clean     # remove generated PDFs

PANDOC      := pandoc
PDF_ENGINE  := wkhtmltopdf
CSS         := style.css

JP_SRC      := D12_pre_registration_OSF.md
EN_SRC      := D12_pre_registration_OSF.en.md
JP_PDF      := D12_pre_registration_OSF.pdf
EN_PDF      := D12_pre_registration_OSF.en.pdf

PANDOC_FLAGS := --pdf-engine=$(PDF_ENGINE) --css=$(CSS)

.PHONY: all jp en clean

all: jp en

jp: $(JP_PDF)
en: $(EN_PDF)

$(JP_PDF): $(JP_SRC) $(CSS)
	$(PANDOC) $(JP_SRC) -o $(JP_PDF) $(PANDOC_FLAGS) \
	  --metadata title="D12 OSF Pre-Registration (Japanese master)"

$(EN_PDF): $(EN_SRC) $(CSS)
	$(PANDOC) $(EN_SRC) -o $(EN_PDF) $(PANDOC_FLAGS) \
	  --metadata title="D12 OSF Pre-Registration (English version)"

clean:
	rm -f $(JP_PDF) $(EN_PDF)
