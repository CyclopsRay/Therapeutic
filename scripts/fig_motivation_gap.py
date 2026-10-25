# -*- coding: utf-8 -*-
"""The coverage gap: what a perturbation screen reaches vs where the answer is needed."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY, GRAY, ORANGE = '#102D43', '#56636C', '#EC633D'
MUTED = '#8FA3B0'
W, H = 11.98, 2.90

fig = plt.figure(figsize=(W, H), dpi=220)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

def pill(x, y, w, h, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0,rounding_size=0.14', **kw))

# ---- section labels
ax.text(0.35, 2.66, 'CAN SCREEN', color=NAVY, fontsize=11.5, fontweight='bold')
ax.text(3.95, 2.66, 'CANNOT SCREEN  —  and this is where the answer is needed',
        color=ORANGE, fontsize=11.5, fontweight='bold')

# ---- what a screen reaches
pill(0.35, 0.80, 2.75, 1.62, facecolor=NAVY, edgecolor='none')
ax.text(1.725, 1.85, '50', color='white', fontsize=30, fontweight='bold',
        ha='center', va='center')
ax.text(1.725, 1.40, 'cancer cell lines', color='white', fontsize=13,
        ha='center', va='center')
ax.text(1.725, 1.07, 'in a dish  ·  24 h', color=MUTED, fontsize=11,
        ha='center', va='center')

# ---- what it does not reach
LABELS = ['Patient tumours', 'Rare cell types', 'Won’t grow ex vivo',
          'Primary cells', 'Immune subsets', '… thousands more']
xs = [3.95, 6.68, 9.41]
ys = [1.70, 0.90]
for i, lab in enumerate(LABELS):
    x, y = xs[i % 3], ys[i // 3]
    pill(x, y, 2.45, 0.62, facecolor='white', edgecolor=ORANGE,
         linewidth=1.3, linestyle=(0, (4, 3)))
    ax.text(x + 1.225, y + 0.31, lab, color=NAVY, fontsize=11.5,
            ha='center', va='center')

# ---- the barrier
ax.add_patch(FancyArrowPatch((3.25, 1.61), (3.80, 1.61), arrowstyle='-|>',
                             mutation_scale=17, linewidth=2.0, color=ORANGE))
ax.text(3.95, 0.42,
        'Weeks to months and a wet lab for each one — many can never be screened at all',
        color=GRAY, fontsize=12, va='center')

fig.savefig('fig/motivation_gap.png', dpi=220, facecolor='white')
print('ok')
