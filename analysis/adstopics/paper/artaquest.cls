%% artaquest.cls — the official document class for Journal of Seasonality.
%% A clean, single-column (two-column optional), MATHS-first article class in the journal's
%% two-colour brand (gold #E8B923 · the why, blue #2352E8 · the how). Times-like text +
%% matching maths, generous margins, a running header, numbered clickable references,
%% cleveref cross-refs (equation/figure/table/section AND theorem-like envs), brand-styled
%% theorem environments, a documented maths helper set, and a reusable pgfplots "house
%% style" for figures that survive greyscale + colour-blindness. Plus a structured title
%% block + end-matter. Drop it next to your main.tex (Overleaf: "Upload Project" the zip).
%% Standard CTAN packages only; compiles with tectonic or pdflatex.
%%
%%   Class options:
%%     [review]     line numbers + double spacing (handy while drafting / for review)
%%     [twocolumn]  two-column body (single column is the default)
%%     any article option (a4paper, letterpaper, 10pt, 11pt, 12pt …)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{artaquest}[2026/06/27 v3.0 Journal of Seasonality article class]

\newif\ifaq@review\aq@reviewfalse
\DeclareOption{review}{\aq@reviewtrue}
% Pass any unknown option straight to article (e.g. a4paper, 12pt, twocolumn).
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt]{article}

% ── Page geometry — generous, readable margins. ──────────────────────────────────
\RequirePackage[a4paper,margin=1in,headsep=16pt,footskip=28pt]{geometry}

% ── Mathematics: load BEFORE the maths font (newtxmath) so symbols resolve cleanly. ─
% amsmath → amssymb → mathtools → amsthm, then the matched maths font. mathtools is a
% strict superset of amsmath and supplies \DeclarePairedDelimiter, \coloneqq, etc.
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{mathtools}
\RequirePackage{amsthm}
\allowdisplaybreaks   % long aligned displays may break across a page rather than overflow

% ── Fonts: Times-like text + matching maths (academic standard), graceful fallback. ─
\RequirePackage[T1]{fontenc}
\RequirePackage{textcomp}
\IfFileExists{newtxtext.sty}{%
  \RequirePackage{newtxtext}%
  \IfFileExists{newtxmath.sty}{\RequirePackage[varvw]{newtxmath}}{}%
}{%
  \RequirePackage{lmodern}% fallback when newtx is unavailable
}
\RequirePackage{microtype}

% ── Core typesetting packages. ───────────────────────────────────────────────────
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{titlesec}
\RequirePackage{titling}        % gives us \thetitle/\theauthor for the running header
\RequirePackage{enumitem}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{subcaption}
\RequirePackage[font=small,labelfont={bf,sc,color=aqblue},labelsep=period,
                margin=1em,justification=justified,singlelinecheck=false,skip=6pt,
                hangindent=1em]{caption}
\RequirePackage{siunitx}
\sisetup{detect-all,group-separator={,},group-minimum-digits=4}
\RequirePackage{listings}
\RequirePackage{fancyhdr}
\RequirePackage[numbers,sort&compress,square,comma]{natbib}
\bibliographystyle{plainnat}    % numbered references; overridable in main.tex

% ── Visualisation engine: tikz + pgfplots (loaded so authors needn't). ───────────
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{patterns}
\usepgfplotslibrary{groupplots}

\RequirePackage[colorlinks=true,unicode]{hyperref}
\RequirePackage[nameinlink,capitalise]{cleveref} % MUST be after hyperref + amsmath

% A \doi{…} helper for the bibliography (clickable DOI links in references.bib).
\providecommand{\doi}[1]{\href{https://doi.org/#1}{doi:#1}}

% Let long URLs (in availability statements / references) break gracefully rather than
% overrun the measure — allow a break after any of these characters, not only "/".
\renewcommand{\UrlBreaks}{\do\.\do\@\do\\\do\/\do\!\do\_\do\|\do\;\do\>\do\]%
  \do\)\do\,\do\?\do\'\do\+\do\=\do\#\do\-\do\&}
\Urlmuskip=0mu plus 1mu\relax

% ── Brand: gold (the why) + blue (the how). The only two accents. ────────────────
% Plus light tints + neutral greys DERIVED from the brand (no third hue) — used by the
% figure house style so series stay distinguishable in colour, greyscale and for
% colour-blind readers (colour is never the only channel: dash + marker also vary).
\definecolor{aqgold}{HTML}{E8B923}
\definecolor{aqblue}{HTML}{2352E8}
\definecolor{aqink}{HTML}{16181D}
\definecolor{aqgrey}{HTML}{5B6066}
\definecolor{aqgoldlight}{HTML}{F4D77A}   % light gold (secondary warm series)
\definecolor{aqbluelight}{HTML}{7E97F2}   % light blue (secondary cool series)
\definecolor{aqgreydark}{HTML}{3A3E44}    % near-ink grey (reference lines / muted)
\hypersetup{
  linkcolor=aqblue, citecolor=aqblue, urlcolor=aqblue, breaklinks=true,
  pdfborderstyle={/S/U/W 0},
  pdftitle={\@title}, pdfsubject={Journal of Seasonality}, pdfcreator={artaquest.cls}}

% ── Comfortable reading measure + tight widow/orphan control. ────────────────────
\linespread{1.05}
\setlength{\parskip}{0.2\baselineskip}
\setlength{\parindent}{0pt}
\setlist{leftmargin=1.4em,itemsep=0.2em,topsep=0.3em}
\frenchspacing
\clubpenalty=10000          % no orphan (first line of a paragraph alone at page foot)
\widowpenalty=10000         % no widow (last line of a paragraph alone at page top)
\displaywidowpenalty=10000
\predisplaypenalty=10000
\brokenpenalty=10000        % avoid breaking a page right after a hyphenated line

% ── Float defaults — keep floats near their reference, no orphaned pages. ─────────
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.7}
\setlength{\textfloatsep}{16pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{6pt}

%% ===================================================================================
%%  MATHEMATICS — a documented helper set + brand-styled theorem environments.
%% ===================================================================================
% Right-numbered displays as (1) are amsmath's default; we keep them. \allowdisplaybreaks
% (set above) lets long aligned displays break across pages.

% — Operators. \DeclareMathOperator gives upright, correctly-spaced names. —
\DeclareMathOperator*{\argmin}{arg\,min}   % starred → limits go underneath in display
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator{\Var}{Var}            % variance
\DeclareMathOperator{\Cov}{Cov}            % covariance
\DeclareMathOperator{\Corr}{Corr}          % correlation
\DeclareMathOperator{\E}{\mathbb{E}}       % expectation:  \E[X]
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\sign}{sign}

% — Common scalar statistics, typeset upright (a metric is a name, not a product). —
\newcommand{\RMSE}{\mathrm{RMSE}}
\newcommand{\MAE}{\mathrm{MAE}}
\newcommand{\MAPE}{\mathrm{MAPE}}
\newcommand{\Rsq}{R^{2}}                    % coefficient of determination

% — Paired delimiters: \abs{x}, \norm{v} auto-size; the * star forces single size. —
\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}
\DeclarePairedDelimiter{\norm}{\lVert}{\rVert}
\DeclarePairedDelimiter{\set}{\{}{\}}
\DeclarePairedDelimiter{\ceil}{\lceil}{\rceil}
\DeclarePairedDelimiter{\floor}{\lfloor}{\rfloor}

% — Vectors (bold lowercase) + matrices (bold uppercase). Consistent, semantic. —
\newcommand{\vect}[1]{\boldsymbol{#1}}      % \vect{x}  → a vector
\newcommand{\mat}[1]{\mathbf{#1}}           % \mat{A}   → a matrix
\newcommand{\T}{^{\mathsf{T}}}              % transpose: \mat{A}\T

% — Number sets. —
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}

% mathtools convenience: := as \coloneqq, ≈ etc. are already provided; expose a couple.
\providecommand{\dd}{\mathop{}\!\mathrm{d}}  % upright differential:  \int f(x)\dd x

% — Brand-styled theorem-like environments. ―――――――――――――――――――――――――――――――――――――
% A coloured small-caps run-in head; numbered within the section. "Plain" style
% (italic body) for results; "definition" style (upright body) for definitions/remarks.
\newtheoremstyle{aqplain}%
  {0.7em}{0.7em}%                     space above / below
  {\itshape}%                          body font
  {0pt}%                               indent
  {\sffamily\bfseries\color{aqblue}}%  head font (blue · the how)
  {.}{0.6em}%                          punctuation after head; space after head
  {}%                                  default head spec
\newtheoremstyle{aqdefn}%
  {0.7em}{0.7em}%
  {\normalfont}%                       upright body
  {0pt}%
  {\sffamily\bfseries\color{aqgold!72!black}}% head font (gold · the why)
  {.}{0.6em}{}
\theoremstyle{aqplain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{aqdefn}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{assumption}[theorem]{Assumption}
% A QED square in brand gold for the amsthm proof environment.
\renewcommand{\qedsymbol}{\textcolor{aqgold}{$\blacksquare$}}

% cleveref names for the theorem-like environments (singular / plural).
\crefname{theorem}{Theorem}{Theorems}
\crefname{lemma}{Lemma}{Lemmas}
\crefname{proposition}{Proposition}{Propositions}
\crefname{corollary}{Corollary}{Corollaries}
\crefname{definition}{Definition}{Definitions}
\crefname{remark}{Remark}{Remarks}
\crefname{assumption}{Assumption}{Assumptions}
\crefname{equation}{Equation}{Equations}

%% ===================================================================================
%%  VISUALISATION — the "aq" pgfplots house style. Apply with [aq] to every axis:
%%     \begin{axis}[aq, xlabel=…, ylabel=…] … \end{axis}
%%  Colour-blind-safe by construction: each series differs by COLOUR *and* DASH *and*
%%  MARKER, so figures survive greyscale and colour-vision deficiency. Light grid, tidy
%%  legend, sensible default size, no chartjunk.
%% ===================================================================================
\pgfplotsset{
  % The reusable axis style authors apply to every plot.
  aq/.style={
    width=0.86\linewidth, height=5.4cm,
    font=\footnotesize,
    label style={font=\small\color{aqink}},
    tick label style={font=\footnotesize\color{aqgrey}},
    title style={font=\small\bfseries\color{aqink}},
    every axis plot/.append style={line width=0.9pt},
    axis line style={aqgrey!70, line width=0.6pt},
    axis on top,                       % gridlines never paint over the data
    tick align=outside, tick style={aqgrey!60, thin},
    grid=major, grid style={aqgrey!16, line width=0.4pt},
    enlarge x limits=0.02, enlarge y limits=0.04,
    legend cell align=left,
    legend style={
      font=\footnotesize, draw=aqgrey!35, fill=white, fill opacity=0.92,
      text opacity=1, rounded corners=2pt, inner sep=4pt, row sep=1pt,
      legend image post style={line width=0.9pt}},
    % Colour-blind-safe cycle: colour + dash + marker all change per series.
    cycle list={
      {aqblue,      solid,            mark=*,        mark size=1.6pt, mark options={fill=aqblue,draw=white,line width=0.3pt}},
      {aqgold!85!black, dashed,       mark=square*,  mark size=1.6pt, mark options={fill=aqgold!85!black,draw=white,line width=0.3pt}},
      {aqgreydark,  densely dotted,   mark=triangle*,mark size=2.0pt, mark options={fill=aqgreydark,draw=white,line width=0.3pt}},
      {aqbluelight, dash dot,         mark=diamond*, mark size=1.9pt, mark options={fill=aqbluelight,draw=white,line width=0.3pt}},
      {aqgoldlight, loosely dashed,   mark=pentagon*,mark size=2.0pt, mark options={fill=aqgoldlight,draw=white,line width=0.3pt}},
    },
  },
  % A smooth, no-marker variant for analytic curves / dense series.
  aq line/.style={aq, no markers, every axis plot/.append style={line width=1.0pt}},
  % Per-plot styles, applied to an \addplot, not the axis: ───────────────────────
  % the exemplar / median / "answer" curve — heavier brand blue, no dashes.
  aq emphasis/.style={aqblue, line width=1.6pt, mark=none, smooth},
  % a secondary / comparison series — muted grey, dashed, light.
  aq muted/.style={aqgreydark!70, line width=0.9pt, dashed, mark=none},
  % a warm secondary series in gold.
  aq warm/.style={aqgold!88!black, line width=1.1pt, mark=none},
  % a horizontal/vertical reference line (a baseline, a threshold, a target).
  aq rule/.style={aqgrey!75, line width=0.7pt, densely dashed, mark=none},
  % a filled confidence/spread band: \addplot[aq band] fill between two curves.
  aq band/.style={draw=none, fill=aqblue!12},
}

% ── Captions: bold small-caps "Figure N." in gold-ish blue, hanging, self-contained. ─
% (caption package configured above; nothing more needed here — the labelfont carries
%  the bf+sc+colour, labelsep=period gives "Figure 1.", hangindent aligns wrapped text.)

% ── Code listings — brand-styled (a gold left rule), for reproducible snippets. ───
\lstdefinestyle{aq}{
  basicstyle=\ttfamily\footnotesize, keywordstyle=\color{aqblue}\bfseries,
  commentstyle=\color{aqgrey}\itshape, stringstyle=\color{aqgold!75!black},
  numbers=left, numberstyle=\tiny\color{aqgrey}, numbersep=8pt,
  frame=leftline, framerule=2pt, rulecolor=\color{aqgold}, xleftmargin=1.2em,
  breaklines=true, breakatwhitespace=false, postbreak=\mbox{\textcolor{aqgrey}{$\hookrightarrow$}\space},
  showstringspaces=false, columns=fullflexible, keepspaces=true,
  backgroundcolor=\color{aqblue!3}, aboveskip=0.8em, belowskip=0.4em}
\lstset{style=aq}

% ── Section headings — numbered, brand blue; tight, confident spacing. ───────────
\titleformat{\section}{\normalfont\large\bfseries\color{aqblue}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries\color{aqblue}}{\thesubsection}{0.5em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\itshape\color{aqink}}{\thesubsubsection}{0.5em}{}
\titleformat{\paragraph}[runin]{\normalfont\bfseries\color{aqink}}{}{0pt}{}[.\;]
\titlespacing*{\section}{0pt}{1.6ex plus .2ex}{0.7ex}
\titlespacing*{\subsection}{0pt}{1.2ex plus .2ex}{0.5ex}
\titlespacing*{\subsubsection}{0pt}{1.0ex plus .2ex}{0.4ex}

% ── Running header (journal · short title · page) + a licence footer on page 1. ──
% Optional short running title for the header; defaults to the full \title.
\newcommand{\aq@runtitle}{}
\newcommand{\runningtitle}[1]{\renewcommand{\aq@runtitle}{#1}}
\newcommand{\aq@headtitle}{\ifx\aq@runtitle\empty\thetitle\else\aq@runtitle\fi}

\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}\renewcommand{\footrulewidth}{0pt}
\fancyhead[L]{\footnotesize\color{aqgrey}Journal of Seasonality}
\fancyhead[C]{\footnotesize\itshape\color{aqgrey}\aq@headtitle}
\fancyhead[R]{\footnotesize\color{aqgrey}\thepage}
\fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}%
  \fancyfoot[L]{\footnotesize\color{aqgrey}Journal of Seasonality \textbullet\ open access \textbullet\ CC\,BY\,4.0}%
  \fancyfoot[R]{\footnotesize\color{aqgrey}\thepage}}

% ── Title-block metadata. ────────────────────────────────────────────────────────
\newcommand{\aq@affiliation}{}
\newcommand{\affiliation}[1]{\renewcommand{\aq@affiliation}{#1}}
\newcommand{\aq@keywords}{}
\newcommand{\keywords}[1]{\renewcommand{\aq@keywords}{#1}}

% Author/affiliation superscript marker → a small raised number, used both after an
% author name and at the head of each affiliation line. Render the digits at a fixed
% small size and raise them (don't rely on \textsuperscript alone: with newtx, wrapping
% the argument in \normalfont disables the font's superior figures and the marker drops
% to the baseline — the bug this macro exists to avoid):
%   \author{Alice\affmark{1}\orcid{0000-…} \and Bob\affmark{2}}
%   \affiliation{\affmark{1}Institute A \quad \affmark{2}Institute B}
\newcommand{\affmark}[1]{\raisebox{0.55ex}{\scriptsize#1}}
% ORCID helper: \orcid{0000-0000-0000-0000} → a small, recognisable "iD" badge link.
% A gold rounded chip carrying white "iD" letters (the ORCID glyph, recoloured to the
% brand gold — never a third hue), drawn with tikz at superscript height so it sits neatly
% after the author name like the marker on a journal byline.
\DeclareRobustCommand{\orcid}[1]{\,\href{https://orcid.org/#1}{%
  \raisebox{0.35ex}{\begin{tikzpicture}[baseline=(c.base)]%
    \node[fill=aqgold,rounded corners=1.3pt,inner sep=0pt,minimum size=1.45ex,
          font=\sffamily\bfseries\fontsize{4.6pt}{4.6pt}\selectfont,text=white]
      (c) {iD};%
  \end{tikzpicture}}}}

% Render the title block. Call \maketitle after \title/\author/\affiliation, in document.
% The block ends with ONE thin gold rule; the abstract follows directly (no second rule,
% no empty gap). Author line slightly larger; affiliation small italic grey; date small.
\renewcommand{\maketitle}{%
  \thispagestyle{plain}%
  \begingroup\renewcommand{\and}{\unskip\hspace{1.5em}}% authors may use \and; space them along the line
  % Masthead: brand label left, access statement right, then ONE bold gold rule.
  {\footnotesize\color{aqblue}\textsc{Journal of Seasonality}\hfill
    \footnotesize\color{aqgrey}Open access \textbullet\ CC\,BY\,4.0\par}
  \vspace{0.3em}{\color{aqgold}\rule{\linewidth}{1.6pt}}\par\vspace{1.0em}
  % Title — large, tightly leaded, ink. The headline of the page.
  {\raggedright\bfseries\fontsize{21}{25}\selectfont\color{aqink}\@title\par}
  \vspace{0.85em}
  % Authors — a touch larger than body, with the ORCID badges riding the names.
  {\raggedright\large\color{aqink}\@author\par}
  \ifx\aq@affiliation\empty\else
    \vspace{0.35em}{\raggedright\footnotesize\itshape\color{aqgrey}\aq@affiliation\par}\fi
  \vspace{0.45em}
  {\raggedright\footnotesize\color{aqgrey}\@date\par}
  \endgroup
  \vspace{0.85em}{\color{aqgold}\rule{\linewidth}{0.4pt}}\par
  \vspace{0.75em}
}

% ── Abstract: an upright roman lead block (flagship-journal practice — roman reads more
% comfortably than a long italic run), set slightly smaller and justified for a distinct,
% inviting opening; a blue small-caps "Abstract" run-in head; keywords styled beneath it;
% one closing gold rule. (The opening rule is the single one drawn by \maketitle, so the
% masthead → abstract is one clean rule, not two with a gap.)
\renewenvironment{abstract}{%
  \begingroup\small\linespread{1.04}\selectfont
  \noindent{\sffamily\bfseries\color{aqblue}Abstract\quad}\ignorespaces
}{%
  \ifx\aq@keywords\empty\else\par\vspace{0.55em}%
    \noindent{\footnotesize\color{aqink}%
      {\sffamily\bfseries\color{aqblue}Keywords}\enspace\textemdash\enspace\aq@keywords}\fi
  \par\endgroup
  \vspace{0.55em}{\color{aqgold}\rule{\linewidth}{0.4pt}}\par\vspace{0.7em}
}

% ── Structured end-matter. ───────────────────────────────────────────────────────
% Data/Code availability are collected and printed by \declarations; the rest are
% inline declaration macros you call after it.
\newcommand{\aq@data}{}
\newcommand{\dataavailability}[1]{\renewcommand{\aq@data}{#1}}
\newcommand{\aq@code}{}
\newcommand{\codeavailability}[1]{\renewcommand{\aq@code}{#1}}

% End-matter declarations often hold long URLs; typeset them loosely so a long link
% wraps rather than overruns the measure (\sloppy is scoped to the paragraph).
\newcommand{\aq@decl}[2]{\par\begingroup\sloppy\noindent
  {\sffamily\bfseries\color{aqink}#1.}\ #2\par\endgroup\vspace{0.3em}}
\newcommand{\declarations}{%
  \section*{Declarations}%
  \ifx\aq@data\empty\else\aq@decl{Data availability}{\aq@data}\fi
  \ifx\aq@code\empty\else\aq@decl{Code availability}{\aq@code}\fi
}
\newcommand{\authorcontributions}[1]{\aq@decl{Author contributions}{#1}}
\newcommand{\competinginterests}[1]{\aq@decl{Competing interests}{#1}}
\newcommand{\funding}[1]{\aq@decl{Funding}{#1}}
\newcommand{\acknowledgements}[1]{\aq@decl{Acknowledgements}{#1}}
\newcommand{\acknowledgments}[1]{\acknowledgements{#1}}% US-spelling alias

% Tidy the bibliography heading + spacing.
\renewcommand{\bibsection}{\section*{\refname}}
\setlength{\bibsep}{0.4ex plus 0.2ex}

% ── [review] option → double spacing + line numbers for reviewer/author reference. ─
\ifaq@review
  \RequirePackage{setspace}\AtBeginDocument{\doublespacing}
  \RequirePackage[mathlines]{lineno}\AtBeginDocument{\linenumbers}
\fi

\endinput
