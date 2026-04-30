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
DIFF_SRC    := D12_pre_registration_v1.1_to_v2.0_diff.md
CLAR_SRC    := D12_v2.0_methods_clarifications.md
JP_PDF      := D12_pre_registration_OSF.pdf
EN_PDF      := D12_pre_registration_OSF.en.pdf
DIFF_PDF    := D12_pre_registration_v1.1_to_v2.0_diff.pdf
CLAR_PDF    := D12_v2.0_methods_clarifications.pdf

PANDOC_FLAGS := --pdf-engine=$(PDF_ENGINE) --css=$(CSS)

.PHONY: all jp en diff clarifications clean

all: jp en diff clarifications

jp: $(JP_PDF)
en: $(EN_PDF)
diff: $(DIFF_PDF)
clarifications: $(CLAR_PDF)

$(JP_PDF): $(JP_SRC) $(CSS)
	$(PANDOC) $(JP_SRC) -o $(JP_PDF) $(PANDOC_FLAGS) \
	  --metadata title="D12 OSF Pre-Registration (Japanese master)"

$(EN_PDF): $(EN_SRC) $(CSS)
	$(PANDOC) $(EN_SRC) -o $(EN_PDF) $(PANDOC_FLAGS) \
	  --metadata title="D12 OSF Pre-Registration (English version)"

$(DIFF_PDF): $(DIFF_SRC) $(CSS)
	$(PANDOC) $(DIFF_SRC) -o $(DIFF_PDF) $(PANDOC_FLAGS) \
	  --metadata title="D12 Pre-Registration Public Diff v1.1 to v2.0"

$(CLAR_PDF): $(CLAR_SRC) $(CSS)
	$(PANDOC) $(CLAR_SRC) -o $(CLAR_PDF) $(PANDOC_FLAGS) \
	  --metadata title="D12 v2.0 Methods Clarifications Log (Section 6.5 Level 1)"

clean:
	rm -f $(JP_PDF) $(EN_PDF) $(DIFF_PDF) $(CLAR_PDF)
