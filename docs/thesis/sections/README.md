# Текст на тезата — готови `.tex` датотеки

Секоја датотека е **веќе конвертирана во чист LaTeX**, подготвена за директно копирање во
Overleaf. Нема повеќе Markdown-синтакса (`**`, `#`, `|...|` табели) — сè е `\textbf{}`,
`\chapter{}`/`\section{}`, `\begin{tabular}` итн.

## Мапа: датотека → место во трудот

| Датотека | Каде оди | Состојба |
|---|---|---|
| `01-abstract-mk.tex` | страница „Апстракт“ | ✅ готово |
| `02-abstract-en.tex` | страница „Abstract“ | ✅ готово |
| `03-acronyms.tex` | страница „Акроними“ | ✅ готово — целосна замена на туѓиот список |
| `04-chapter2-problem-formulation.tex` | Глава 2 — Формулација на проблемот | ✅ готово |
| `05-chapter3-theoretical-foundations.tex` | Глава 3 — Теоретски основи (ново поглавје) | ✅ готово |
| `06-chapter4-related-work.tex` | Глава 4 — Поврзани истражувања | ✅ готово |
| `07-methodology.tex` | Глава 5 — Методологија | ✅ веќе средена во Overleaf; **не се менува** |
| `08-chapter6-architecture.tex` | Глава 6 — Архитектурна логика | ✅ готово |
| `09-chapter7-results.tex` | Глава 7 — Експерименти и резултати | ✅ готово |
| `10-chapter8-limitations.tex` | Глава 8 — Ограничувања | ✅ готово |
| `11-chapter9-conclusion.tex` | Глава 9 — Заклучок и дискусија | ✅ готово |
| `12-chapter10-future-work.tex` | Глава 10 — Идни планови | ✅ готово |
| `../references.bib` | библиографија | ✅ 20 потврдени извори |

Имињата на датотеките се намерно на латиница (ASCII) — некои LaTeX-алатки паѓаат на
кириличен `\input{}`-пат.

## Структура од 10 поглавја

```
1  Вовед                          (веќе напишано, надвор од sections/)
2  Формулација на проблемот       04-chapter2-problem-formulation.tex
3  Теоретски основи               05-chapter3-theoretical-foundations.tex  ← НОВО ПОГЛАВЈЕ
4  Поврзани истражувања           06-chapter4-related-work.tex
5  Методологија                   07-methodology.tex
6  Архитектурна логика            08-chapter6-architecture.tex
7  Експерименти и резултати       09-chapter7-results.tex
8  Ограничувања                   10-chapter8-limitations.tex
9  Заклучок и дискусија           11-chapter9-conclusion.tex
10 Идни планови                   12-chapter10-future-work.tex
```

## LaTeX-ознаки (сите користат конвенцијата `chap:` — усогласено со `07-methodology.tex`)

`\label{chap:problem}` · `\label{chap:theory}` · `\label{chap:related}` ·
`\label{chap:methodology}` · `\label{chap:system}` · `\label{chap:results}` ·
`\label{chap:limitations}` · `\label{chap:conclusion}` · `\label{chap:future}`

Табели вклучени со `\input{}` (генерирани со `python -m analysis.make_figures`):
`tables/tab_paired_cost.tex` (`\label{tab:paired-cost}`) и `tables/tab_utility.tex`
(`\label{tab:utility}`). Овие двете **не се дефинираат тука** — ознаките ќе се решат штом
ги генерирате и вклучите табелите.

## ⚠️ Уште потребно пред компајлирање

1. **`\label{fig:h6}`** — Глава 7 повикува `\ref{fig:h6}` (сликата за трошокот на
   сопственост). Додајте ја таа ознака кога ја вметнувате самата слика во Overleaf.
2. **Фотографија на хардверот** — во `07-methodology.tex`, `\includegraphics` е
   закоментиран (потписот е готов, недостасува само сликата).
3. **Регенерирајте ги табелите** пред да ги пастирате: `python -m analysis.make_figures`
   ги гради `docs/thesis/tables/tab_paired_cost.tex` и `tab_utility.tex`.

## Проверено

- LaTeX-балансот (`itemize`/`enumerate`/`quote`/`table`/`tabular`) е чист во сите датотеки.
- Сите `\cite{}` клучеви постојат во `references.bib` (20/20), ниту еден неупотребен.
- Сите `\ref{}`/`\hyperref[]` повикувања покажуваат кон постоечка ознака, освен трите
  веќе-документирани исклучоци погоре.
- Секоја бројка во Глава 7 (најгустата со податоци) е поединечно проверена наспроти
  `results/msc/runs/*.csv`.

## Работни белешки (надвор од оваа папка, не се пастираат во Overleaf)

- `docs/thesis-notes.md` — лабораториски дневник од пишувањето
- `docs/thesis/00-поимник-работна-белешка.md` — поимник МК↔EN и правила за стил
- `docs/thesis/07-методологија-исправки-применети.md` — историски список на седумте
  исправки веќе применети во `07-methodology.tex`
