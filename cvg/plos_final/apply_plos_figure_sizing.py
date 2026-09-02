import re, pathlib, sys
src=pathlib.Path("/home/svc-jax-dlh/.work/python_project/src/analysis.py").read_text()
t=src

def sub1(old,new):
    global t
    assert t.count(old)==1, f"count={t.count(old)}: {old[:60]}"
    t=t.replace(old,new)

# ---------------- transfer_matrix ----------------
sub1("figure_width, figure_height = 18.0, 8.8",
     "figure_width, figure_height = 9.30, 7.10")
sub1("    group_header_height = 1.3\n    header_height = 2 * group_header_height",
     "    group_header_height = 1.5\n    header_height = group_header_height + 5.4")
sub1("source_left, decoder_left = -3.10, -1.85",
     "source_left, decoder_left = -4.15, -2.35")
# vertical target-column headers: they were the element that destroyed the layout
sub1("""            column + 0.5, label_header_top / 2, label, ha="center", va="center",
            fontsize=9, fontweight="bold", clip_on=False,""",
     """            column + 0.5, label_header_top / 2, label, ha="center", va="center",
            rotation=90, fontsize=8, fontweight="bold", clip_on=False,""")
# fonts into the PLOS 8-12 pt band (canvas is now final print size)
sub1("""                fontsize=8.4,
                fontweight="bold" if not np.isnan(value) else "normal",""",
     """                fontsize=8.0,
                fontweight="bold" if not np.isnan(value) else "normal",""")
sub1("""            label, ha="center", va="center", fontsize=9, fontweight="bold",
            clip_on=False,
        )
        for decoder_index, model_name in enumerate(model_names):""",
     """            label, ha="center", va="center", fontsize=8, fontweight="bold",
            clip_on=False,
        )
        for decoder_index, model_name in enumerate(model_names):""")
sub1("""                -0.08, y_position + decoder_index + 0.5, model_name,
                ha="right", va="center", fontsize=9, fontweight="bold",""",
     """                -0.08, y_position + decoder_index + 0.5, model_name,
                ha="right", va="center", fontsize=8, fontweight="bold",""")
sub1("""            group_label, ha="center", va="center", fontsize=9, fontweight="bold",""",
     """            group_label, ha="center", va="center", fontsize=9, fontweight="bold",""")
sub1("""        va="center", fontsize=11, fontweight="bold", clip_on=False,
    )
    ax.text(
        source_left - 0.28,""",
     """        va="center", fontsize=10, fontweight="bold", clip_on=False,
    )
    ax.text(
        source_left - 0.28,""")
sub1("""        va="center", rotation=90, fontsize=11, fontweight="bold", clip_on=False,""",
     """        va="center", rotation=90, fontsize=10, fontweight="bold", clip_on=False,""")
sub1("""    colorbar.set_label(f"{ARTICLE_METRIC_LABEL}, %", fontsize=9, fontweight="bold")""",
     """    colorbar.set_label(f"{ARTICLE_METRIC_LABEL}, %", fontsize=8, fontweight="bold")""")

# ---------------- scenario_accuracy_by_decoder ----------------
sub1("figsize=(12.2, 12.9)", "figsize=(7.5, 8.2)")
sub1("""    ax.text(0.5, 1.0, "chance level", color="#62718A", ha="center", va="bottom", fontsize=10,""",
     """    ax.text(0.5, 1.0, "chance level", color="#62718A", ha="center", va="bottom", fontsize=8,""")
sub1("""    ax.set_xlabel(ARTICLE_METRIC_LABEL, fontsize=12, fontweight="bold", color="#24354F")
    ax.set_yticks""",
     """    ax.set_xlabel(ARTICLE_METRIC_LABEL, fontsize=10, fontweight="bold", color="#24354F")
    ax.set_yticks""")
sub1("""                fontsize=11 if label in protocol_label_set else 10)""",
     """                fontsize=9 if label in protocol_label_set else 8)""")
sub1("""    legend = ax.legend(
        ncols=6,
        loc="upper center",
        bbox_to_anchor=(0.37, 1.054),
        frameon=False,
        fontsize=11,
        markerscale=1.15,
        handletextpad=0.35,
        columnspacing=0.75,
    )""",
     """    legend = ax.legend(
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.30, 1.075),
        frameon=False,
        fontsize=8,
        markerscale=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
    )""")
sub1("""    fig.subplots_adjust(left=0.22, right=0.98, top=0.9385, bottom=0.048)""",
     """    fig.subplots_adjust(left=0.30, right=0.985, top=0.930, bottom=0.058)""")

# ---------------- baseline_vs_cross_subject ----------------
sub1("figsize=(11.4, 5.6)", "figsize=(7.5, 3.9)")
sub1("""        ax.set_title(title, fontsize=15, fontweight="bold", color="#172B4D", loc="left", pad=12)""",
     """        ax.set_title(title, fontsize=10, fontweight="bold", color="#172B4D", loc="left", pad=8)""")
sub1("""        ax.set_xticks(x_positions, [label for _, label in protocol_specs], fontsize=12,""",
     """        ax.set_xticks(x_positions, [label for _, label in protocol_specs], fontsize=9,""")
sub1("""        ax.tick_params(axis="both", length=0, labelsize=11)""",
     """        ax.tick_params(axis="both", length=0, labelsize=8)""")
sub1("""    axes[0].set_ylabel(ARTICLE_METRIC_LABEL, fontsize=12, fontweight="bold", color="#24354F")""",
     """    axes[0].set_ylabel(ARTICLE_METRIC_LABEL, fontsize=10, fontweight="bold", color="#24354F")""")
sub1("""        ncols=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=11,""",
     """        ncols=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=8,""")
sub1("""    fig.subplots_adjust(left=0.055, right=0.985, top=0.865, bottom=0.145, wspace=0.07)""",
     """    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.235, wspace=0.07)""")

# summary header labels collided: narrower wording, smaller type, wider columns
sub1("""    summary_left, summary_width = 0.62, 0.13""",
     """    summary_left, summary_width = 0.625, 0.155""")
sub1("""        "Baseline\\nvs\\ncross-task\\nmean diff",
        "Baseline\\nvs\\ncross-dataset\\nmean diff",""",
     """        "Baseline vs\\ncross-task\\nmean diff",
        "Baseline vs\\ncross-\\ndataset\\nmean diff",""")
sub1("""            column + 0.5, -header_height / 2, label, ha="center", va="center",
            fontsize=9, fontweight="bold", clip_on=False,""",
     """            column + 0.5, -header_height / 2, label, ha="center", va="center",
            fontsize=7.5, fontweight="bold", clip_on=False,""")
sub1("""                fontsize=11,
                fontweight="bold" if not np.isnan(value) else "normal",
                color="#263341",""",
     """                fontsize=10,
                fontweight="bold" if not np.isnan(value) else "normal",
                color="#263341",""")

pathlib.Path(sys.argv[1]).write_text(t)
print("patched ok")
