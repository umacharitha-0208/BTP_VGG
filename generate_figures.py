"""Generate APEX architecture figures."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 180,
})

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, color, fontsize=10, textcolor='black', style='round,pad=0.1', lw=1.5, sublabel=None):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=style, linewidth=lw,
                       edgecolor='#333333', facecolor=color, zorder=3)
    ax.add_patch(b)
    if sublabel:
        ax.text(x, y + 0.02, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=textcolor, zorder=4)
        ax.text(x, y - 0.06, sublabel, ha='center', va='center',
                fontsize=fontsize-1.5, color=textcolor, zorder=4, style='italic')
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=textcolor, zorder=4)

def arr(ax, x1, y1, x2, y2, color='#555555', lw=1.8, label='', fontsize=8.5,
        connectionstyle="arc3,rad=0.0", arrowstyle='->', lbl_offset=(0,0.04)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, connectionstyle=connectionstyle),
                zorder=2)
    if label:
        mx = (x1+x2)/2 + lbl_offset[0]
        my = (y1+y2)/2 + lbl_offset[1]
        ax.text(mx, my, label, ha='center', va='bottom',
                fontsize=fontsize, color=color, zorder=5)

# ═════════════════════════════════════════════════════════════
# FIGURE 1 — High-level overview
# ═════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(13, 5.5))
ax1.set_xlim(0, 13); ax1.set_ylim(0, 5.5)
ax1.axis('off')
ax1.set_facecolor('white')

# Image boxes
for y, lbl in [(3.9, 'I'), (1.6, "I'")]:
    img_box = FancyBboxPatch((0.2, y-0.7), 1.4, 1.4, boxstyle='round,pad=0.05',
                              linewidth=1.5, edgecolor='#888', facecolor='#dde8f5')
    ax1.add_patch(img_box)
    ax1.text(0.9, y, lbl, ha='center', va='center', fontsize=16,
             fontweight='bold', color='#333', zorder=4)

ax1.text(0.9, 0.6, 'Input: Unlabeled\nimage pairs only',
         ha='center', va='center', fontsize=8.5, color='#555', style='italic')

# Encoder-decoder + PPO actor-critic blocks
for y, tag in [(3.9, ''), (1.6, "'")]:
    # encoder
    b = FancyBboxPatch((2.0, y-0.55), 0.85, 1.1, boxstyle='round,pad=0.05',
                        linewidth=1.5, edgecolor='#333', facecolor='#c8daf5')
    ax1.add_patch(b)
    ax1.text(2.425, y+0.15, r'$\mathbf{e}_\theta$', ha='center', va='center',
             fontsize=13, color='#222', zorder=4)

    # decoder
    b2 = FancyBboxPatch((3.0, y-0.55), 0.85, 1.1, boxstyle='round,pad=0.05',
                         linewidth=1.5, edgecolor='#333', facecolor='#b8e8c8')
    ax1.add_patch(b2)
    ax1.text(3.425, y+0.15, r'$\mathbf{d}_\theta$', ha='center', va='center',
             fontsize=13, color='#222', zorder=4)

    # PPO actor-critic
    b3 = FancyBboxPatch((4.0, y-0.55), 1.3, 1.1, boxstyle='round,pad=0.05',
                         linewidth=1.8, edgecolor='#8b1a1a', facecolor='#ffe4b5')
    ax1.add_patch(b3)
    ax1.text(4.65, y+0.18, 'Actor-Critic', ha='center', va='center',
             fontsize=8.5, fontweight='bold', color='#8b1a1a', zorder=4)
    ax1.text(4.65, y-0.08, '(PPO Policy)', ha='center', va='center',
             fontsize=7.5, color='#8b1a1a', style='italic', zorder=4)

    # arrows image→encoder→decoder→ppo
    arr(ax1, 1.65, y, 2.0, y, color='#e67e00', lw=2)
    arr(ax1, 2.88, y, 3.0, y, color='#555')
    arr(ax1, 3.88, y, 4.0, y, color='#555')

# Labels on outputs
for y, tag in [(3.9, ''), (1.6, "'")]:
    ax1.text(5.45, y+0.3, 'Location', ha='left', va='center',
             fontsize=8.5, color='#2a7a2a')
    ax1.text(5.45, y-0.25, 'Descriptor', ha='left', va='center',
             fontsize=8.5, color='#1a50a0')
    ax1.text(5.45, y+0.0, 'Value', ha='left', va='center',
             fontsize=8.5, color='#8b1a1a')
    arr(ax1, 5.3, y+0.3, 5.45, y+0.3, color='#2a7a2a', lw=1.2)
    arr(ax1, 5.3, y-0.25, 5.45, y-0.25, color='#1a50a0', lw=1.2)
    arr(ax1, 5.3, y+0.0, 5.45, y+0.0, color='#8b1a1a', lw=1.2)

# MNN + LO-RANSAC box
b_mnn = FancyBboxPatch((6.4, 2.0), 1.7, 1.5, boxstyle='round,pad=0.1',
                         linewidth=2, edgecolor='#333', facecolor='#f0f0f0')
ax1.add_patch(b_mnn)
ax1.text(7.25, 2.95, 'MNN', ha='center', va='center',
         fontsize=11, fontweight='bold', color='#333', zorder=4)
ax1.text(7.25, 2.65, '+', ha='center', va='center',
         fontsize=11, color='#333', zorder=4)
ax1.text(7.25, 2.35, 'LO-RANSAC', ha='center', va='center',
         fontsize=10, fontweight='bold', color='#333', zorder=4)

arr(ax1, 6.1, 3.9+0.3, 6.4, 2.8, color='#2a7a2a', lw=1.5,
    connectionstyle='arc3,rad=0.15', label='Location', lbl_offset=(-0.3, 0.05))
arr(ax1, 6.1, 1.6+0.3, 6.4, 2.2, color='#2a7a2a', lw=1.5,
    connectionstyle='arc3,rad=-0.15')
arr(ax1, 6.1, 3.9-0.25, 7.25, 2.0+1.5, color='#1a50a0', lw=1.5,
    connectionstyle='arc3,rad=0.1', label='Descriptors', lbl_offset=(0.5, 0.1))
arr(ax1, 6.1, 1.6-0.25, 7.25, 2.0, color='#1a50a0', lw=1.5,
    connectionstyle='arc3,rad=-0.1')

# PPO reward box
b_ppo = FancyBboxPatch((9.4, 2.15), 2.2, 1.2, boxstyle='round,pad=0.1',
                         linewidth=2.5, edgecolor='#8b1a1a', facecolor='#fff0e0')
ax1.add_patch(b_ppo)
ax1.text(10.5, 2.95, 'PPO Update', ha='center', va='center',
         fontsize=11, fontweight='bold', color='#8b1a1a', zorder=4)
ax1.text(10.5, 2.6, r'$L^{CLIP} + L^{VF} + L^{InfoNCE}$', ha='center', va='center',
         fontsize=8, color='#8b1a1a', zorder=4)

arr(ax1, 9.1, 2.75, 9.4, 2.75, color='#8b1a1a', lw=2.5, label='Reward')

# Gradient arrows back
ax1.annotate('', xy=(2.425, 3.9+0.55), xytext=(10.5, 3.35),
             arrowprops=dict(arrowstyle='->', color='#cc3333', lw=2.5,
                             connectionstyle='arc3,rad=-0.25'), zorder=2)
ax1.annotate('', xy=(2.425, 1.6-0.55), xytext=(10.5, 2.15),
             arrowprops=dict(arrowstyle='->', color='#cc3333', lw=2.5,
                             connectionstyle='arc3,rad=0.25'), zorder=2)
ax1.text(7.5, 4.9, 'Gradient', ha='center', va='center',
         fontsize=9, color='#cc3333', fontweight='bold')
ax1.text(7.5, 0.5, 'Gradient', ha='center', va='center',
         fontsize=9, color='#cc3333', fontweight='bold')

# Value feedback arrows
arr(ax1, 6.1, 3.9+0.0, 9.4, 2.85, color='#8b1a1a', lw=1.2,
    connectionstyle='arc3,rad=-0.2', label='Value  ', lbl_offset=(0.0, 0.15))
arr(ax1, 6.1, 1.6+0.0, 9.4, 2.65, color='#8b1a1a', lw=1.2,
    connectionstyle='arc3,rad=0.2')

ax1.set_title('APEX — High-Level Training Overview', fontsize=13, fontweight='bold', pad=10)
fig1.tight_layout()
fig1.savefig(r'C:\Users\linga\Downloads\BTP_MODIFIED_VGG\fig1_apex_overview.png',
             dpi=180, bbox_inches='tight', facecolor='white')
print('Figure 1 saved.')
plt.close(fig1)

# ═════════════════════════════════════════════════════════════
# FIGURE 2 — Detection + Description detail
# ═════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(13, 7.5))
ax2.set_xlim(0, 13); ax2.set_ylim(0, 7.5)
ax2.axis('off')

# ── DETECTION (top) ──────────────────────────────────────────
det_bg = FancyBboxPatch((0.3, 3.8), 12.4, 3.4, boxstyle='round,pad=0.15',
                          linewidth=2, edgecolor='#e0a000', facecolor='#fffbe6')
ax2.add_patch(det_bg)
ax2.text(6.5, 7.0, 'Detection', ha='center', va='center',
         fontsize=13, fontweight='bold', color='#b07000')

# Image
img_d = FancyBboxPatch((0.5, 4.5), 1.2, 1.5, boxstyle='round,pad=0.05',
                         linewidth=1.5, edgecolor='#888', facecolor='#dde8f5')
ax2.add_patch(img_d)
ax2.text(1.1, 5.25, 'I', ha='center', va='center', fontsize=16,
         fontweight='bold', color='#333')

# encoder / decoder
b_e = FancyBboxPatch((2.0, 4.6), 0.9, 1.3, boxstyle='round,pad=0.05',
                       linewidth=1.5, edgecolor='#333', facecolor='#c8daf5')
ax2.add_patch(b_e)
ax2.text(2.45, 5.25, r'$\mathbf{e}_\theta$', ha='center', va='center',
         fontsize=13, color='#222')

b_d = FancyBboxPatch((3.1, 4.6), 0.9, 1.3, boxstyle='round,pad=0.05',
                       linewidth=1.5, edgecolor='#333', facecolor='#b8e8c8')
ax2.add_patch(b_d)
ax2.text(3.55, 5.25, r'$\mathbf{d}_\theta$', ha='center', va='center',
         fontsize=13, color='#222')

arr(ax2, 1.72, 5.25, 2.0, 5.25, color='#e67e00', lw=2)
arr(ax2, 2.92, 5.25, 3.1, 5.25, color='#555')

# Heatmap grid H
ax2.text(4.65, 6.7, r'$\mathbf{H}$', ha='center', va='center',
         fontsize=13, fontweight='bold', color='#333')
ax2.text(5.7, 6.7, r'$s_i$', ha='center', va='center',
         fontsize=11, color='#333', style='italic')
ax2.text(6.4, 6.75, r'$c_i$', ha='center', va='center',
         fontsize=11, color='#333', style='italic')
ax2.annotate('', xy=(6.35, 6.55), xytext=(6.0, 6.75),
             arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))

# Draw grid
grid_x, grid_y0 = 4.2, 4.1
cell = 0.55
colors_grid = [
    ['#ff8c00','#4caf50','#ff8c00','#4caf50'],
    ['#4caf50','#ff8c00','#4caf50','#ff8c00'],
    ['#ff8c00','#4caf50','#ff8c00','#4caf50'],
    ['#4caf50','#ff8c00','#4caf50','#4caf50'],
]
for r in range(4):
    for c in range(4):
        fc = colors_grid[r][c]
        rect = plt.Rectangle((grid_x + c*cell, grid_y0 + r*cell), cell, cell,
                               linewidth=1, edgecolor='#aaa', facecolor=fc, alpha=0.7, zorder=3)
        ax2.add_patch(rect)
        if r == 3 and c == 3:
            hatch = plt.Rectangle((grid_x + c*cell, grid_y0 + r*cell), cell, cell,
                                    linewidth=1, edgecolor='#aaa', facecolor='none',
                                    hatch='///', zorder=4)
            ax2.add_patch(hatch)

arr(ax2, 4.02, 5.25, 4.2, 5.25, color='#555', lw=1.5)

# Actor-Critic PPO block
b_ppo2 = FancyBboxPatch((6.55, 4.55), 1.5, 1.35, boxstyle='round,pad=0.07',
                          linewidth=2, edgecolor='#8b1a1a', facecolor='#ffe4b5')
ax2.add_patch(b_ppo2)
ax2.text(7.3, 5.4, 'PPO', ha='center', va='center',
         fontsize=10, fontweight='bold', color='#8b1a1a')
ax2.text(7.3, 5.1, 'Actor-Critic', ha='center', va='center',
         fontsize=8.5, color='#8b1a1a')

arr(ax2, 6.4, 5.25, 6.55, 5.25, color='#555', lw=1.5)

# Keypoint score S
b_s = FancyBboxPatch((8.3, 4.65), 0.45, 1.15, boxstyle='round,pad=0.05',
                       linewidth=1.5, edgecolor='#333', facecolor='#4caf50', alpha=0.8)
ax2.add_patch(b_s)
ax2.text(8.52, 5.22, r'$\mathbf{S}$', ha='center', va='center',
         fontsize=13, fontweight='bold', color='white')

arr(ax2, 8.05, 5.22, 8.3, 5.22, color='#555', lw=1.5)
ax2.text(6.5, 4.25, 'Keypoint Locations', ha='center', va='center',
         fontsize=9, color='#555', style='italic')
ax2.text(4.3, 3.92, 'Encoder Layer', ha='left', va='center',
         fontsize=9, color='#555', style='italic')

# ── DESCRIPTION (bottom) ──────────────────────────────────────
desc_bg = FancyBboxPatch((0.3, 0.3), 12.4, 3.2, boxstyle='round,pad=0.15',
                           linewidth=2, edgecolor='#1a50a0', facecolor='#eef4ff')
ax2.add_patch(desc_bg)
ax2.text(6.5, 3.3, 'Description', ha='center', va='center',
         fontsize=13, fontweight='bold', color='#1a50a0')

# Multi-scale features (colored bars)
bar_colors = ['#2ecc71', '#e74c3c', '#9b59b6']
bar_heights = [2.0, 1.4, 0.8]
bar_x0 = 0.8
for i, (bh, bc) in enumerate(zip(bar_heights, bar_colors)):
    rect = plt.Rectangle((bar_x0, 1.5 - bh/2), 0.45, bh,
                           linewidth=1.2, edgecolor='#333', facecolor=bc, alpha=0.8, zorder=3)
    ax2.add_patch(rect)
    bar_x0 += 0.55

# Hypercolumn stacked bars D_hyper
ax2.text(3.2, 2.9, r'$\mathbf{D}_{hyper}$', ha='center', va='center',
         fontsize=11, fontweight='bold', color='#333')
for ky in [2.55, 2.1, 1.65, 1.2]:
    for kxi, (bw, bc) in enumerate(zip([0.6, 0.4, 0.25], bar_colors)):
        kx = 2.5 + kxi*bw
        rect = plt.Rectangle((kx, ky-0.14), bw, 0.28,
                               linewidth=0.8, edgecolor='#555', facecolor=bc, alpha=0.75, zorder=3)
        ax2.add_patch(rect)
ax2.text(3.2, 0.9, '...', ha='center', va='center', fontsize=14, color='#555')

arr(ax2, 2.15, 1.5, 2.5, 1.9, color='#555', lw=1.2, connectionstyle='arc3,rad=-0.1')

# ATTENTION HEAD box (replaces 1x1 Conv)
b_attn = FancyBboxPatch((4.8, 1.0), 2.2, 2.0, boxstyle='round,pad=0.1',
                          linewidth=2.5, edgecolor='#8b1a1a', facecolor='#fff0e0')
ax2.add_patch(b_attn)
ax2.text(5.9, 2.5, 'Attention', ha='center', va='center',
         fontsize=10, fontweight='bold', color='#8b1a1a')
ax2.text(5.9, 2.2, 'Descriptor Head', ha='center', va='center',
         fontsize=9, fontweight='bold', color='#8b1a1a')
ax2.text(5.9, 1.85, 'Linear → MHA(4h)', ha='center', va='center',
         fontsize=8, color='#8b1a1a', style='italic')
ax2.text(5.9, 1.58, '+ LayerNorm + FFN', ha='center', va='center',
         fontsize=8, color='#8b1a1a', style='italic')
ax2.text(5.9, 1.25, '→ L2-norm', ha='center', va='center',
         fontsize=8, color='#8b1a1a', style='italic')

arr(ax2, 4.45, 1.9, 4.8, 1.9, color='#555', lw=1.5)

# Descriptor output D
b_desc = FancyBboxPatch((7.3, 1.1), 0.45, 1.8, boxstyle='round,pad=0.05',
                          linewidth=1.5, edgecolor='#333', facecolor='#4caf50', alpha=0.8)
ax2.add_patch(b_desc)
ax2.text(7.52, 2.0, r'$\mathbf{D}$', ha='center', va='center',
         fontsize=13, fontweight='bold', color='white')
for ky in [1.7, 2.0, 2.3]:
    ax2.plot([7.3, 7.75], [ky, ky], color='white', lw=0.8, zorder=4)

arr(ax2, 7.0, 1.9, 7.3, 1.9, color='#555', lw=1.5)

# InfoNCE loss annotation
b_loss = FancyBboxPatch((8.5, 1.3), 3.8, 1.3, boxstyle='round,pad=0.1',
                          linewidth=1.8, edgecolor='#1a50a0', facecolor='#ddeeff')
ax2.add_patch(b_loss)
ax2.text(10.4, 2.2, 'InfoNCE Contrastive Loss', ha='center', va='center',
         fontsize=9.5, fontweight='bold', color='#1a50a0')
ax2.text(10.4, 1.85, r'$\mathcal{L}_{InfoNCE} = -\log\frac{\exp(z_i \cdot z_j / \tau)}{\sum_k \exp(z_i \cdot z_k / \tau)}$',
         ha='center', va='center', fontsize=8.5, color='#1a50a0')

arr(ax2, 7.77, 2.0, 8.5, 1.9, color='#1a50a0', lw=1.5)

# Keypoint location arrow from top to bottom
ax2.annotate('', xy=(4.65, 2.55), xytext=(4.65, 4.1),
             arrowprops=dict(arrowstyle='->', color='#2a7a2a', lw=1.5,
                             connectionstyle='arc3,rad=0.0'), zorder=2)
ax2.text(4.85, 3.3, 'Keypoint\nLocations', ha='left', va='center',
         fontsize=8, color='#2a7a2a')

ax2.set_title('APEX — Model Architecture Detail', fontsize=13, fontweight='bold', pad=10)
fig2.tight_layout()
fig2.savefig(r'C:\Users\linga\Downloads\BTP_MODIFIED_VGG\fig2_apex_architecture.png',
             dpi=180, bbox_inches='tight', facecolor='white')
print('Figure 2 saved.')
plt.close(fig2)

# ═════════════════════════════════════════════════════════════
# FIGURE 3 — Full training loop
# ═════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(15, 6))
ax3.set_xlim(0, 15); ax3.set_ylim(0, 6)
ax3.axis('off')

# λ=1 box
b_lam = FancyBboxPatch((0.3, 2.6), 0.9, 0.65, boxstyle='round,pad=0.07',
                         linewidth=1.2, edgecolor='#555', facecolor='#f5f5f5')
ax3.add_patch(b_lam)
ax3.text(0.75, 2.92, r'$\lambda=1$', ha='center', va='center',
         fontsize=10, color='#333')

# Image boxes
for y, lbl in [(4.5, 'I'), (1.5, "I'")]:
    ib = FancyBboxPatch((0.3, y-0.7), 1.3, 1.4, boxstyle='round,pad=0.05',
                          linewidth=1.5, edgecolor='#888', facecolor='#dde8f5')
    ax3.add_patch(ib)
    ax3.text(0.95, y, lbl, ha='center', va='center',
             fontsize=16, fontweight='bold', color='#333')

arr(ax3, 0.75, 2.6, 0.75, 2.2, color='#555', lw=1.2)
arr(ax3, 0.75, 3.25, 0.75, 3.55, color='#555', lw=1.2)

# Detect + Describe blocks
for y in [4.5, 1.5]:
    b = FancyBboxPatch((1.9, y-0.7), 1.8, 1.4, boxstyle='round,pad=0.08',
                         linewidth=2, edgecolor='#2a7a2a', facecolor='#e8f5e9')
    ax3.add_patch(b)
    ax3.text(2.8, y+0.15, 'Detect +', ha='center', va='center',
             fontsize=9.5, fontweight='bold', color='#2a7a2a')
    ax3.text(2.8, y-0.15, 'Describe', ha='center', va='center',
             fontsize=9.5, fontweight='bold', color='#2a7a2a')
    arr(ax3, 1.62, y, 1.9, y, color='#e67e00', lw=2)

# Output labels
outputs_top    = [('P', 4.95, '#8b1a1a', 'Probability (Actor)'),
                  ('S', 4.65, '#2a7a2a', 'Location'),
                  ('a', 4.35, '#555555', 'Acceptance'),
                  ('V', 4.05, '#8b1a1a', 'Value (Critic)'),
                  ('D', 3.75, '#1a50a0', 'Descriptor')]
outputs_bot    = [("P'", 1.05, '#8b1a1a', ''),
                  ("S'", 1.35, '#2a7a2a', ''),
                  ("a'", 1.65, '#555555', ''),
                  ("V'", 1.95, '#8b1a1a', ''),
                  ("D'", 2.25, '#1a50a0', '')]

for sym, y, col, lbl in outputs_top:
    arr(ax3, 3.72, y, 3.92, y, color=col, lw=1.2)
    ax3.text(3.95, y, sym, ha='left', va='center',
             fontsize=10, fontweight='bold', color=col)
    if lbl:
        ax3.text(4.35, y, lbl, ha='left', va='center',
                 fontsize=8, color=col)

for sym, y, col, _ in outputs_bot:
    arr(ax3, 3.72, y, 3.92, y, color=col, lw=1.2)
    ax3.text(3.95, y, sym, ha='left', va='center',
             fontsize=10, fontweight='bold', color=col)

# MNN box
b_mnn3 = FancyBboxPatch((6.0, 2.5), 1.8, 1.5, boxstyle='round,pad=0.1',
                           linewidth=2, edgecolor='#333', facecolor='#f0f0f0')
ax3.add_patch(b_mnn3)
ax3.text(6.9, 3.42, 'Mutual Nearest', ha='center', va='center',
         fontsize=8.5, fontweight='bold', color='#333')
ax3.text(6.9, 3.15, 'Neighbor', ha='center', va='center',
         fontsize=8.5, fontweight='bold', color='#333')
ax3.text(6.9, 2.85, 'Matching', ha='center', va='center',
         fontsize=8.5, fontweight='bold', color='#333')

# LO-RANSAC box
b_rans = FancyBboxPatch((8.2, 2.5), 1.6, 1.5, boxstyle='round,pad=0.1',
                           linewidth=2, edgecolor='#333', facecolor='#f0f0f0')
ax3.add_patch(b_rans)
ax3.text(9.0, 3.35, 'LO-RANSAC', ha='center', va='center',
         fontsize=9, fontweight='bold', color='#333')
ax3.text(9.0, 3.05, '(PROSAC)', ha='center', va='center',
         fontsize=8, color='#555', style='italic')
ax3.text(9.0, 2.75, '+ epipolar', ha='center', va='center',
         fontsize=8, color='#555', style='italic')

arr(ax3, 7.8, 3.25, 8.2, 3.25, color='#555', lw=1.8)

# Inlier reward grid R
for rr in range(3):
    for cc in range(3):
        fc = '#4caf50' if (rr+cc) % 2 == 0 else 'white'
        rect = plt.Rectangle((10.1 + cc*0.28, 2.55 + rr*0.28), 0.28, 0.28,
                               linewidth=0.8, edgecolor='#333', facecolor=fc, alpha=0.8, zorder=3)
        ax3.add_patch(rect)
        if fc == '#4caf50':
            ax3.text(10.1 + cc*0.28 + 0.14, 2.55 + rr*0.28 + 0.14,
                     '+', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax3.text(10.52, 2.38, r'$\mathbf{R}$ (Inliers)', ha='center', va='center',
         fontsize=9, color='#333')
arr(ax3, 9.8, 3.25, 10.1, 3.15, color='#555', lw=1.5)

# Log probs box
b_lp = FancyBboxPatch((10.1, 3.65), 1.7, 0.8, boxstyle='round,pad=0.07',
                         linewidth=1.5, edgecolor='#8b1a1a', facecolor='#fff0e0')
ax3.add_patch(b_lp)
ax3.text(10.95, 4.15, r'$\log \pi_\theta(a|s)$', ha='center', va='center',
         fontsize=8.5, color='#8b1a1a')
ax3.text(10.95, 3.88, r'$\log \pi_{\theta_{old}}(a|s)$', ha='center', va='center',
         fontsize=8.5, color='#8b1a1a')

# Advantage box
b_adv = FancyBboxPatch((10.1, 1.55), 1.7, 0.7, boxstyle='round,pad=0.07',
                         linewidth=1.5, edgecolor='#8b1a1a', facecolor='#fff0e0')
ax3.add_patch(b_adv)
ax3.text(10.95, 1.9, r'$\hat{A}_t = R_t - V(s_t)$', ha='center', va='center',
         fontsize=8.5, color='#8b1a1a')

# PPO gradient box
b_grad = FancyBboxPatch((12.0, 2.3), 2.7, 1.8, boxstyle='round,pad=0.1',
                           linewidth=2.5, edgecolor='#8b1a1a', facecolor='#ffe9d0')
ax3.add_patch(b_grad)
ax3.text(13.35, 3.85, 'PPO Gradient', ha='center', va='center',
         fontsize=10, fontweight='bold', color='#8b1a1a')
ax3.text(13.35, 3.5,  r'$r_t = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$',
         ha='center', va='center', fontsize=9, color='#8b1a1a')
ax3.text(13.35, 3.1,
         r'$L^{CLIP} = \mathbb{E}[\min(r_t\hat{A}_t,$',
         ha='center', va='center', fontsize=8, color='#8b1a1a')
ax3.text(13.35, 2.78,
         r'$\mathrm{clip}(r_t, 1\pm\varepsilon)\hat{A}_t)]$',
         ha='center', va='center', fontsize=8, color='#8b1a1a')
ax3.text(13.35, 2.48, r'$+ \mathcal{L}_{InfoNCE} + \beta H(\pi_\theta)$',
         ha='center', va='center', fontsize=8, color='#1a50a0')

# sum symbol
ax3.text(11.55, 3.28, r'$\bigoplus$', ha='center', va='center',
         fontsize=18, color='#333', zorder=5)
arr(ax3, 10.85, 3.1, 11.4, 3.28, color='#555', lw=1.2)
arr(ax3, 10.85, 4.05, 11.4, 3.38, color='#8b1a1a', lw=1.2)
arr(ax3, 10.85, 1.9, 11.4, 3.15, color='#8b1a1a', lw=1.2)
arr(ax3, 11.72, 3.28, 12.0, 3.28, color='#555', lw=1.5)

# S, D, a connector arrows to MNN/RANSAC
arr(ax3, 4.35+0.3, 4.65, 6.9, 4.0,  color='#2a7a2a', lw=1.3,
    connectionstyle='arc3,rad=0.2')
arr(ax3, 4.35+0.3, 1.35, 6.9, 2.5,  color='#2a7a2a', lw=1.3,
    connectionstyle='arc3,rad=-0.2')
arr(ax3, 4.35+0.3, 3.75, 6.9, 2.5+1.5, color='#1a50a0', lw=1.3,
    connectionstyle='arc3,rad=0.15', label='D', lbl_offset=(-0.2, 0.08))
arr(ax3, 4.35+0.3, 2.25, 6.9, 2.5,  color='#1a50a0', lw=1.3,
    connectionstyle='arc3,rad=-0.15')

# Gradient feedback arrow
ax3.annotate('', xy=(2.8, 5.2), xytext=(13.35, 4.1),
             arrowprops=dict(arrowstyle='->', color='#cc3333', lw=2.5,
                             connectionstyle='arc3,rad=-0.3'), zorder=2)
ax3.text(8.5, 5.55, 'Gradient (PPO update)', ha='center', va='center',
         fontsize=9.5, color='#cc3333', fontweight='bold')

ax3.set_title('APEX — Full Training Loop', fontsize=13, fontweight='bold', pad=8)
fig3.tight_layout()
fig3.savefig(r'C:\Users\linga\Downloads\BTP_MODIFIED_VGG\fig3_apex_training_loop.png',
             dpi=180, bbox_inches='tight', facecolor='white')
print('Figure 3 saved.')
plt.close(fig3)

print('\nAll 3 figures saved to C:\\Users\\linga\\Downloads\\BTP_MODIFIED_VGG\\')
