"""AI-Based UPI Spending Analyzer  -  Desktop app (PySide6 / Qt).

Same AI engine as the web version (analyzer/ package), shown in a native window.
Run:   python desktop_app.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QColor, QIcon, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFileDialog, QTabWidget, QTableView, QHeaderView, QSpinBox, QLineEdit,
    QFrame, QMessageBox, QProgressBar, QScrollArea, QSplashScreen, QComboBox, QTextEdit,
    QSizePolicy,
)

import analyzer as A
from analyzer import insights, parser, categorizer, llm

APP_TITLE = "AI-Based UPI Spending Analyzer"
ACCENT = "#5b2a86"          # UPI-ish purple
PALETTE = ["#5b2a86", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#264653",
           "#8ab17d", "#b5838d", "#6d6875", "#457b9d", "#a8dadc"]

STYLE = f"""
QMainWindow, QWidget {{ background: #f6f5fa; font-family: 'Segoe UI', 'Noto Sans', sans-serif; font-size: 10pt; }}
#Sidebar {{ background: #ffffff; border-right: 1px solid #e3e0ec; }}
#Sidebar QLabel#H {{ font-weight: 600; color: {ACCENT}; font-size: 10.5pt; margin-top: 8px; }}
QPushButton {{ background: {ACCENT}; color: white; border: none; border-radius: 6px; padding: 8px 12px; }}
QPushButton:hover {{ background: #6f3aa0; }}
QPushButton#Flat {{ background: #ece7f5; color: {ACCENT}; }}
QSpinBox, QLineEdit, QComboBox {{ background: white; border: 1px solid #d6d1e3; border-radius: 5px; padding: 5px; }}
#Card {{ background: white; border: 1px solid #e3e0ec; border-radius: 10px; }}
#CardValue {{ font-size: 17pt; font-weight: 700; color: #1d1b26; }}
#CardLabel {{ color: #6b6780; font-size: 9pt; }}
#Title {{ font-size: 18pt; font-weight: 700; color: #1d1b26; }}
#Sub {{ color: #6b6780; }}
QTabWidget::pane {{ border: 1px solid #e3e0ec; border-radius: 8px; background: white; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 8px 14px; color: #55516a; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 3px solid {ACCENT}; font-weight: 600; }}
QTableView {{ background: white; border: none; gridline-color: #eeeaf5; selection-background-color: #e6dcf3; selection-color: black; }}
QHeaderView::section {{ background: #f1edf8; padding: 5px; border: none; font-weight: 600; }}
QProgressBar {{ border: 1px solid #d6d1e3; border-radius: 5px; background: white; text-align: center; height: 18px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
"""


def rs(x):
    return f"₹{x:,.0f}"


# ---------------------------------------------------------------- helpers
class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, money_cols=(), pct_cols=(), risk_col=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.money, self.pct, self.risk_col = set(money_cols), set(pct_cols), risk_col

    def rowCount(self, parent=QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.df.columns)

    def data(self, index, role=Qt.DisplayRole):
        col = self.df.columns[index.column()]
        v = self.df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if col in self.money and pd.notna(v):
                return rs(float(v))
            if col in self.pct and pd.notna(v):
                return f"{float(v):.0%}"
            if isinstance(v, pd.Timestamp):
                return v.strftime("%d %b %Y  %I:%M %p")
            if isinstance(v, float):
                return f"{v:,.2f}"
            return str(v)
        if role == Qt.TextAlignmentRole and (col in self.money or col in self.pct):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.BackgroundRole and self.risk_col is not None:
            lvl = self.df.at[index.row(), self.risk_col]
            return QColor({"High": "#fde2e2", "Medium": "#fff1d6", "Low": "#eef6ff"}.get(lvl, "white"))
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return str(self.df.columns[section]).replace("_", " ").title()
        return None


def table(df, **kw):
    v = QTableView()
    v.setModel(DataFrameModel(df, **kw))
    v.verticalHeader().setVisible(False)
    v.setAlternatingRowColors(False)
    v.setSortingEnabled(False)
    v.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    v.horizontalHeader().setStretchLastSection(True)
    return v


def chart(width=6, height=3.6):
    fig = Figure(figsize=(width, height), dpi=100, facecolor="white")
    canvas = FigureCanvasQTAgg(fig)
    canvas.setMinimumHeight(int(height * 85))
    return fig, canvas


def style_ax(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color="#1d1b26")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)


def card(label, value, sub=None, sub_color="#6b6780"):
    f = QFrame(); f.setObjectName("Card")
    lay = QVBoxLayout(f); lay.setContentsMargins(14, 10, 14, 10); lay.setSpacing(2)
    a = QLabel(label); a.setObjectName("CardLabel")
    b = QLabel(value); b.setObjectName("CardValue")
    lay.addWidget(a); lay.addWidget(b)
    if sub:
        c = QLabel(sub); c.setStyleSheet(f"color:{sub_color}; font-size:9pt;")
        lay.addWidget(c)
    return f


def heading(text):
    l = QLabel(text); l.setStyleSheet("font-size:12.5pt; font-weight:700; color:#1d1b26; margin:4px 0;")
    return l


def note(text, bg="#eef4ff", fg="#1d3557"):
    l = QLabel(text); l.setWordWrap(True); l.setTextFormat(Qt.RichText)
    l.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:8px; padding:10px;")
    return l


def scroll(widget):
    s = QScrollArea(); s.setWidgetResizable(True); s.setFrameShape(QFrame.NoFrame); s.setWidget(widget)
    return s


# ---------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.setWindowTitle(APP_TITLE)
        self.resize(1360, 860)
        self.statement = self.load_sample()
        self.source_name = "Built-in sample statement (Aug-Oct 2026)"

        root = QWidget(); self.setCentralWidget(root)
        h = QHBoxLayout(root); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        h.addWidget(self.build_sidebar())
        self.main = QWidget(); self.main_lay = QVBoxLayout(self.main); self.main_lay.setContentsMargins(18, 14, 18, 14)
        h.addWidget(self.main, 1)

        title = QLabel("💸  " + APP_TITLE); title.setObjectName("Title")
        sub = QLabel("FLB project · Unit 5: Current Financial Trends & Fintech · ML categorises, budgets "
                     "and flags risky UPI payments, fully offline"); sub.setObjectName("Sub")
        self.main_lay.addWidget(title); self.main_lay.addWidget(sub)
        self.cards_row = QHBoxLayout(); self.main_lay.addLayout(self.cards_row)
        self.tabs = QTabWidget(); self.main_lay.addWidget(self.tabs, 1)
        self.statusBar().showMessage("Ready")
        self.refresh(first=True)

    # ------------------------------------------------ sidebar
    def build_sidebar(self):
        sb = QWidget(); sb.setObjectName("Sidebar"); sb.setFixedWidth(270)
        v = QVBoxLayout(sb); v.setContentsMargins(16, 16, 16, 16); v.setSpacing(8)
        def H(t):
            l = QLabel(t); l.setObjectName("H"); v.addWidget(l)
        H("1. Statement")
        b = QPushButton("📂  Open UPI statement (CSV / PDF)"); b.clicked.connect(self.open_file); v.addWidget(b)
        b2 = QPushButton("Use sample statement"); b2.setObjectName("Flat"); b2.clicked.connect(self.use_sample); v.addWidget(b2)
        self.src_label = QLabel(); self.src_label.setWordWrap(True); self.src_label.setStyleSheet("color:#6b6780; font-size:9pt;")
        v.addWidget(self.src_label)
        H("2. Budget")
        v.addWidget(QLabel("Monthly income / pocket money (₹)"))
        self.income = QSpinBox(); self.income.setRange(0, 10_000_000); self.income.setSingleStep(500)
        self.income.setGroupSeparatorShown(True); v.addWidget(self.income)
        ab = QPushButton("Apply budget"); ab.setObjectName("Flat"); ab.clicked.connect(lambda: self.refresh(keep_income=True))
        v.addWidget(ab)
        H("3. Optional AI coach")
        self.api_key = QLineEdit(); self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("OpenAI-compatible key (optional)"); v.addWidget(self.api_key)
        tip = QLabel("Leave empty: everything else works fully offline."); tip.setWordWrap(True)
        tip.setStyleSheet("color:#6b6780; font-size:9pt;"); v.addWidget(tip)
        v.addStretch(1)
        self.acc_label = QLabel(); self.acc_label.setWordWrap(True); self.acc_label.setStyleSheet("color:#6b6780; font-size:9pt;")
        v.addWidget(self.acc_label)
        exp = QPushButton("💾  Export categorised CSV"); exp.setObjectName("Flat"); exp.clicked.connect(self.export_csv)
        v.addWidget(exp)
        return sb

    # ------------------------------------------------ data actions
    @staticmethod
    def load_sample():
        return parser.load_statement(str(A.DATA / "sample_upi_statement.csv"), "sample.csv")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open UPI statement", str(Path.home()),
                                              "UPI statements (*.csv *.pdf)")
        if not path:
            return
        try:
            self.statement = parser.load_statement(path, path)
            self.source_name = f"{Path(path).name}"
            self.refresh()
            self.statusBar().showMessage(f"Read {len(self.statement)} transactions from {Path(path).name}", 8000)
        except Exception as e:
            QMessageBox.warning(self, "Could not read file", f"Could not read that file:\n{e}")

    def use_sample(self):
        self.statement = self.load_sample()
        self.source_name = "Built-in sample statement (Aug-Oct 2026)"
        self.refresh()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save categorised CSV", "categorised_upi.csv", "CSV (*.csv)")
        if path:
            self.df.to_csv(path, index=False)
            self.statusBar().showMessage(f"Saved {path}", 6000)

    def refresh(self, first=False, keep_income=False):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.df, self.flagged, self.recurring = A.analyze(self.statement, self.model)
            if not keep_income:
                self.income.setValue(int(insights.detect_monthly_income(self.df) or 20000))
            inc = self.income.value()
            self.budget = insights.budget_503020(self.df, inc)
            self.tips = insights.saving_tips(self.df, self.budget, self.recurring, self.flagged)
            self.k = insights.kpis(self.df)
            self.src_label.setText(f"Now showing: {self.source_name}\n{len(self.df)} transactions")
            self.acc_label.setText(f"Model accuracy: {self.model.accuracy:.1%} on {self.model.n_test} held-out rows")
            self.build_cards()
            current = self.tabs.currentIndex()
            self.tabs.clear()
            for name, fn in [("📊 Overview", self.tab_overview), ("🧾 Transactions", self.tab_transactions),
                             ("🤖 AI Model", self.tab_model), ("🚨 Anomalies & Fraud", self.tab_anomalies),
                             ("🔁 Subscriptions", self.tab_recurring), ("🎯 Budget 50/30/20 & Tips", self.tab_budget),
                             ("📚 FLB Concepts", self.tab_flb)]:
                self.tabs.addTab(fn(), name)
            self.tabs.setCurrentIndex(max(current, 0))
        finally:
            QApplication.restoreOverrideCursor()

    def build_cards(self):
        while self.cards_row.count():
            w = self.cards_row.takeAt(0).widget()
            if w:
                w.deleteLater()
        k, fl = self.k, self.flagged
        high = int((fl["risk_level"] == "High").sum()) if len(fl) else 0
        for c in [card("Total spent", rs(k["total_spent"])), card("Money received", rs(k["total_received"])),
                  card("Avg spend / month", rs(k["avg_monthly_spend"])), card("Transactions", str(k["transactions"])),
                  card("Unusual payments", str(len(fl)), f"{high} high risk", "#c0392b" if high else "#6b6780")]:
            self.cards_row.addWidget(c)

    # ------------------------------------------------ tabs
    def tab_overview(self):
        w = QWidget(); g = QGridLayout(w)
        df = self.df
        # stacked monthly bars
        m = insights.monthly_by_category(df)
        fig, cv = chart(7, 3.6); ax = fig.add_subplot(111)
        labels = [pd.Period(x).strftime("%b %Y") for x in m.index]
        bottom = np.zeros(len(m))
        for i, col in enumerate(m.columns):
            ax.bar(labels, m[col].values, bottom=bottom, label=col, color=PALETTE[i % len(PALETTE)], width=0.55)
            bottom += m[col].values
        style_ax(ax, "Monthly spending by category"); ax.yaxis.set_major_formatter(lambda x, _: f"₹{x/1000:.0f}k")
        ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1), loc="upper left"); fig.tight_layout()
        g.addWidget(cv, 0, 0)
        # donut
        ct = insights.category_totals(df)
        fig2, cv2 = chart(5, 3.6); ax2 = fig2.add_subplot(111)
        ax2.pie(ct.values, labels=None, colors=PALETTE[:len(ct)], startangle=90, counterclock=False,
                wedgeprops=dict(width=0.42, edgecolor="white"), autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
                pctdistance=0.79, textprops=dict(fontsize=8, color="white", fontweight="bold"))
        ax2.legend(ct.index, fontsize=7, frameon=False, bbox_to_anchor=(1.0, 0.5), loc="center left")
        ax2.set_title("Where the money went", loc="left", fontsize=11, fontweight="bold"); fig2.tight_layout()
        g.addWidget(cv2, 0, 1)
        # top merchants
        box = QWidget(); bl = QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(heading("Top merchants"))
        bl.addWidget(table(insights.top_merchants(df), money_cols=["total"]))
        g.addWidget(box, 1, 0)
        # weekly trend
        daily = df[df["type"] == "DEBIT"].set_index("datetime")["amount"].resample("W").sum()
        fig3, cv3 = chart(6, 3.2); ax3 = fig3.add_subplot(111)
        ax3.plot(daily.index, daily.values, marker="o", color=ACCENT, lw=2)
        ax3.fill_between(daily.index, daily.values, alpha=0.08, color=ACCENT)
        style_ax(ax3, "Weekly spending trend"); ax3.yaxis.set_major_formatter(lambda x, _: f"₹{x/1000:.0f}k")
        fig3.autofmt_xdate(); fig3.tight_layout()
        g.addWidget(cv3, 1, 1)
        g.setColumnStretch(0, 3); g.setColumnStretch(1, 2)
        return scroll(w)

    def tab_transactions(self):
        w = QWidget(); v = QVBoxLayout(w)
        top = QHBoxLayout(); top.addWidget(heading("Every transaction, categorised by the model")); top.addStretch(1)
        top.addWidget(QLabel("Category:"))
        combo = QComboBox(); combo.addItem("All"); combo.addItems(sorted(self.df["category"].unique()))
        top.addWidget(combo); v.addLayout(top)
        cols = ["datetime", "merchant", "description", "type", "amount", "category", "confidence", "method"]
        base = self.df[cols].sort_values("datetime", ascending=False)
        view = table(base, money_cols=["amount"], pct_cols=["confidence"]); v.addWidget(view)

        def filt(name):
            d = base if name == "All" else base[base["category"] == name]
            view.setModel(DataFrameModel(d, money_cols=["amount"], pct_cols=["confidence"]))
        combo.currentTextChanged.connect(filt)
        return w

    def tab_model(self):
        md = self.model
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(heading("How the categoriser works"))
        v.addWidget(note(
            "1. <b>Clean</b> the description (remove UPI reference numbers and bank codes).<br>"
            "2. <b>TF-IDF on character n-grams</b> (2-4 letters) turns text into numbers, so SWIGGY, swiggy@ybl and Swigy look alike.<br>"
            "3. <b>Logistic Regression</b> gives a probability for each of the categories.<br>"
            f"4. If the top probability is below <b>{categorizer.CONFIDENCE_THRESHOLD:.0%}</b>, keyword rules decide (fallback). "
            "Money received is always <i>Income</i>."))
        row = QHBoxLayout()
        row.addWidget(card("Test accuracy", f"{md.accuracy:.1%}", "80/20 split"))
        row.addWidget(card("Accuracy on brand-new merchants", f"{md.unseen_accuracy:.1%}", "harder, more honest test"))
        row.addWidget(card("Training / test rows", f"{md.n_train} / {md.n_test}"))
        v.addLayout(row)
        mid = QHBoxLayout()
        fig, cv = chart(6, 4.6); ax = fig.add_subplot(111)
        cm = np.array(md.confusion)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(md.labels))); ax.set_yticks(range(len(md.labels)))
        ax.set_xticklabels(md.labels, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(md.labels, fontsize=7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
        ax.set_title("Confusion matrix (test set)", loc="left", fontsize=11, fontweight="bold"); fig.tight_layout()
        mid.addWidget(cv, 1)
        rep = pd.DataFrame(md.report).T.drop(["accuracy"], errors="ignore")[["precision", "recall", "f1-score", "support"]].round(2)
        rep.insert(0, "category", rep.index)
        rb = QWidget(); rl = QVBoxLayout(rb); rl.addWidget(heading("Per-category precision / recall")); rl.addWidget(table(rep))
        mid.addWidget(rb, 1)
        v.addLayout(mid)
        # live try
        v.addWidget(heading("Try it live"))
        inp = QLineEdit("Paid to Shree Krishna Medical"); v.addWidget(inp)
        bars = QWidget(); bl = QVBoxLayout(bars); bl.setContentsMargins(0, 0, 0, 0); v.addWidget(bars)

        def predict():
            while bl.count():
                it = bl.takeAt(0).widget()
                if it:
                    it.deleteLater()
            for cat, p in categorizer.predict_one(inp.text(), md):
                pb = QProgressBar(); pb.setRange(0, 100); pb.setValue(int(round(p * 100)))
                pb.setFormat(f"{cat}: {p:.0%}"); bl.addWidget(pb)
        inp.textChanged.connect(predict); predict()
        v.addStretch(1)
        return scroll(w)

    def tab_anomalies(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(heading("Payments worth a second look"))
        v.addWidget(QLabel("Duplicate check · z-score on amount · late-night high value · scam words in payee · Isolation Forest"))
        fl = self.flagged
        if fl.empty:
            v.addWidget(note("Nothing unusual found.", "#e7f6ec", "#1e6b3a"))
        else:
            t = table(fl[["risk_level", "datetime", "merchant", "amount", "category", "reasons"]],
                      money_cols=["amount"], risk_col="risk_level")
            t.setMinimumHeight(220); v.addWidget(t)
            deb = self.df[self.df["type"] == "DEBIT"].copy()
            deb["flag"] = deb.index.isin(fl.index)
            hour = deb["datetime"].dt.hour + deb["datetime"].dt.minute / 60
            fig, cv = chart(9, 3.6); ax = fig.add_subplot(111)
            ax.scatter(hour[~deb["flag"]], deb.loc[~deb["flag"], "amount"], s=18, c="steelblue", alpha=0.6, label="normal")
            ax.scatter(hour[deb["flag"]], deb.loc[deb["flag"], "amount"], s=46, c="crimson", label="flagged", edgecolor="white")
            ax.set_yscale("log"); ax.set_xlim(0, 24); ax.set_xlabel("Hour of day", fontsize=8); ax.set_ylabel("₹ (log scale)", fontsize=8)
            style_ax(ax, "Amount vs time of day (red = flagged)"); ax.legend(fontsize=8, frameon=False); fig.tight_layout()
            v.addWidget(cv)
        v.addWidget(note("<b>UPI safety (RBI / NPCI advice):</b> you never need your UPI PIN to <i>receive</i> money; don't approve "
                         "unknown 'collect' requests; no bank asks for KYC updates over UPI; report fraud at 1930 or "
                         "cybercrime.gov.in.", "#fff4e0", "#7a4b00"))
        return scroll(w)

    def tab_recurring(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(heading("Recurring payments & subscriptions"))
        r = self.recurring
        if r.empty:
            v.addWidget(QLabel("No recurring payments found."))
        else:
            row = QHBoxLayout(); row.addWidget(card("Committed every month", rs(r["monthly_cost"].sum()))); row.addStretch(3)
            v.addLayout(row)
            v.addWidget(table(r, money_cols=["typical_amount", "monthly_cost"]))
        v.addWidget(QLabel("Rule: same payee, 3+ payments across 2+ months, roughly monthly, amount varies by 20% or less."))
        v.addStretch(1)
        return w

    def tab_budget(self):
        w = QWidget(); v = QVBoxLayout(w)
        b = self.budget
        v.addWidget(heading(f"50/30/20 budget on {rs(self.income.value())} a month"))
        row = QHBoxLayout()
        fig, cv = chart(6, 3.4); ax = fig.add_subplot(111)
        x = np.arange(len(b))
        ax.bar(x - 0.2, b["target"], 0.4, label="target", color="#c9b6e4")
        ax.bar(x + 0.2, b["actual"], 0.4, label="actual", color=ACCENT)
        ax.set_xticks(x); ax.set_xticklabels(b["bucket"]); ax.axhline(0, color="#999", lw=0.8)
        ax.yaxis.set_major_formatter(lambda x, _: f"₹{x/1000:.0f}k")
        style_ax(ax, "Target vs your average month"); ax.legend(fontsize=8, frameon=False); fig.tight_layout()
        row.addWidget(cv, 3)
        right = QWidget(); rl = QVBoxLayout(right)
        rl.addWidget(table(b, money_cols=["target", "actual"]))
        c = QLabel("Needs: groceries, travel, bills, recharge, education, health.\n"
                   "Wants: eating out, shopping, entertainment, transfers to friends.")
        c.setWordWrap(True); c.setStyleSheet("color:#6b6780; font-size:9pt;"); rl.addWidget(c)
        row.addWidget(right, 2)
        v.addLayout(row)
        v.addWidget(heading("Saving tips for you"))
        v.addWidget(note("<br>".join(f"• {t.replace('Rs ', '₹')}" for t in self.tips), "#eef8f1", "#1e4d2b"))
        ai_btn = QPushButton("✨  Ask the AI coach for a summary (needs API key)")
        out = QTextEdit(); out.setReadOnly(True); out.setVisible(False); out.setMinimumHeight(160)

        def ask():
            key = self.api_key.text().strip()
            if not key:
                QMessageBox.information(self, "AI coach", "Add an API key in the sidebar first. Everything else works offline.")
                return
            k = self.k
            prompt = llm.build_prompt({kk: (round(float(vv)) if isinstance(vv, (int, float)) else vv) for kk, vv in k.items()},
                                      insights.category_totals(self.df).round().to_dict(), b.to_dict("records"),
                                      self.recurring[["merchant", "monthly_cost"]].to_dict("records") if len(self.recurring) else [],
                                      len(self.flagged))
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                out.setMarkdown(llm.ask_llm(prompt, key))
            except Exception as e:
                out.setPlainText(f"AI summary unavailable ({e}). The offline tips above still apply.")
            finally:
                QApplication.restoreOverrideCursor()
            out.setVisible(True)
        ai_btn.clicked.connect(ask)
        h = QHBoxLayout(); h.addWidget(ai_btn); h.addStretch(1); v.addLayout(h); v.addWidget(out)
        v.addStretch(1)
        return scroll(w)

    def tab_flb(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(heading("How each feature maps to the FLB syllabus"))
        v.addWidget(table(pd.DataFrame([
            ("Reading UPI statements (CSV/PDF)", "Digital payments: UPI, NPCI, payment apps", "Unit 5"),
            ("ML categorisation", "Fintech: AI/ML in personal finance apps", "Unit 5"),
            ("Monthly & category charts", "Personal financial planning, tracking expenses", "Budgeting"),
            ("50/30/20 budget", "Budgeting rules: needs vs wants vs savings", "Budgeting"),
            ("Saving tips, pay-yourself-first", "Savings habits, RD / SIP, compounding", "Savings & investment"),
            ("Recurring payments", "Subscription economy, UPI AutoPay / e-mandates", "Unit 5"),
            ("Anomaly & scam detection", "Banking fraud, cyber safety, RBI customer protection", "Unit 3 / fraud awareness"),
        ], columns=["Feature", "FLB concept", "Syllabus link"])))
        v.addWidget(note("<b>Why a desktop app?</b> Bank statements are private. This app runs entirely on your own computer: "
                         "no upload to any server, no internet needed for the ML model (privacy by design, like RBI's "
                         "data-localisation spirit)."))
        return w


# ---------------------------------------------------------------- startup
def make_splash():
    pm = QPixmap(520, 220); pm.fill(QColor(ACCENT))
    p = QPainter(pm); p.setPen(Qt.white)
    f = QFont(); f.setPointSize(18); f.setBold(True); p.setFont(f)
    p.drawText(pm.rect().adjusted(0, -30, 0, 0), Qt.AlignCenter, "💸 UPI Spending Analyzer")
    f.setPointSize(10); f.setBold(False); p.setFont(f)
    p.drawText(pm.rect().adjusted(0, 60, 0, 0), Qt.AlignCenter, "Training the AI model on 3,000 UPI descriptions...")
    p.end()
    return QSplashScreen(pm)


def self_test(out_dir):
    """Used by the Windows build check: open the app, screenshot every tab, read the PDF sample, exit."""
    app = QApplication(sys.argv); app.setStyleSheet(STYLE)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    win = MainWindow(A.load_model()); win.resize(1400, 880); win.show(); app.processEvents()
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i); app.processEvents()
        win.grab().save(str(out / f"tab_{i}.png"))
    win.statement = parser.load_statement(str(A.DATA / "sample_upi_statement.pdf"), "sample.pdf"); win.refresh()
    msg = (f"OK: {len(win.df)} transactions from PDF, {len(win.flagged)} flagged, "
           f"{len(win.recurring)} recurring, accuracy {win.model.accuracy:.3f}")
    (out / "result.txt").write_text(msg, encoding="utf-8")
    print(msg)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        return self_test(sys.argv[2])
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyleSheet(STYLE)
    splash = make_splash(); splash.show(); app.processEvents()
    model = A.load_model()
    win = MainWindow(model)
    win.show(); splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
