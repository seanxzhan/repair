"""Figures for janikova2025-rta-eccentric-joint-strength.md.

All numbers are the paper's per-type means (Tables 2, 3, 6 and 9) and the
dimensions annotated in its Fig. 4(b). Nothing here is simulated.
"""
from _style import *  # noqa: F401,F403
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Circle

TYPES = ["A", "B", "C", "D", "E", "F", "G"]
LABELS = {
    "A": "A  Clamex P-14",
    "B": "B  Clamex + Bisco",
    "C": "C  Tenso P-14 (no glue)",
    "D": "D  Minifix, bolt S100",
    "E": "E  Minifix, M4 + insert nut",
    "F": "F  Minifix, bolt S200",
    "G": "G  Minifix, capped bolt",
}
# Table 2 (compression = closing) and Table 3 (tension = opening): mean M_max (Nm), stiffness (Nm/rad)
M_CLOSE = dict(A=15, B=17, C=5, D=19, E=15, F=18, G=47)
K_CLOSE = dict(A=237, B=208, C=74, D=356, E=233, F=406, G=242)
M_OPEN = dict(A=9, B=11, C=4, D=16, E=14, F=16, G=31)
K_OPEN = dict(A=55, B=62, C=15, D=86, E=59, F=58, G=91)
# Tukey HSD homogeneity groups, Table 6 (closing) and Table 9 (opening)
G_CLOSE = dict(A="1", B="1,2", C="3", D="2", E="1", F="2", G="4")
G_OPEN = dict(A="1", B="1", C="3", D="2", E="4", F="2", G="5")


# ============================================================================
# Figure 1: peak moment and stiffness, closing vs opening, with Tukey groups
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
y = np.arange(len(TYPES))[::-1]

for ax, close, open_, ylabel, title, groups in [
    (axes[0], M_CLOSE, M_OPEN, "peak bending moment $M_{max}$ (Nm)",
     "(a) Strength: G carries 1.9–2.5× the next best;\nevery joint is weaker when opened", True),
    (axes[1], K_CLOSE, K_OPEN, "rotational stiffness (Nm/rad)",
     "(b) Stiffness ranks differently: F and D\nbeat G, and opening is 2.7–7× softer", False),
]:
    c = np.array([close[t] for t in TYPES])
    o = np.array([open_[t] for t in TYPES])
    for yi, ci, oi in zip(y, c, o):
        ax.plot([oi, ci], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.plot(c, y, "o", color=BLUE, ms=7, mec="white", mew=0.8, label="closing (\"compression\")", zorder=3)
    ax.plot(o, y, "o", color=ORANGE, ms=7, mec="white", mew=0.8, label="opening (\"tension\")", zorder=3)
    if groups:
        ax.text(58, y[0] + 0.85, "Tukey group\nclosing / opening", color=INK2, fontsize=7.5, ha="center", va="bottom")
        for yi, t in zip(y, TYPES):
            ax.text(58, yi, f"{G_CLOSE[t]} / {G_OPEN[t]}", color=INK2, fontsize=8, ha="center", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[t] for t in TYPES], fontsize=8.5)
    ax.set_xlabel(ylabel)
    ax.set_xlim(0, None)
    ax.set_title(title, fontsize=9.5)
    clean(ax)
axes[0].set_xlim(0, 66)
axes[0].set_xticks([0, 10, 20, 30, 40, 50])
axes[0].set_ylim(-0.6, len(TYPES) - 0.4 + 1.1)
axes[1].set_xlim(0, 440)
axes[1].set_ylim(-0.6, len(TYPES) - 0.4 + 1.1)
axes[1].set_yticklabels([])
axes[0].legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 1.0))
axes[1].text(0.02, 0.98, "same Tukey group = not\nsignificantly different (95%)", transform=axes[1].transAxes,
             ha="left", va="top", fontsize=7.5, color=INK2)
fig.tight_layout(w_pad=4.0)
save(fig, "janikova2025-rta-eccentric-joint-strength-2.png")  # results


# ============================================================================
# Figure 2: the L-corner test set-up, to scale (Fig. 4(b) of the paper)
# ============================================================================
T = 18.0       # board thickness (mm)
LEG = 150.0    # outer leg length (mm)


def l_corner(angle_deg, apex):
    """Two 18-mm boards meeting at 90 deg with the outer corner at `apex`.

    Returns the two leg polygons. Leg 1 runs along direction angle_deg, leg 2 along
    angle_deg - 90; the leg-2 board butts against the inner face of leg 1
    (leg 1 = face member, leg 2 = rear member butt-joined under it).
    """
    a = np.deg2rad(angle_deg)
    e1 = np.array([np.cos(a), np.sin(a)])
    e2 = np.array([np.cos(a - np.pi / 2), np.sin(a - np.pi / 2)])
    apex = np.asarray(apex, float)
    leg1 = [apex, apex + e1 * LEG, apex + e1 * LEG + e2 * T, apex + e2 * T]
    leg2 = [apex + e2 * T, apex + e2 * T + e1 * T, apex + e2 * LEG + e1 * T, apex + e2 * LEG]
    return np.array(leg1), np.array(leg2), e1, e2


def dim(ax, p, q, text, offset, color=INK2, fs=7.5):
    """Dimension line between p and q, offset perpendicular by `offset` (mm)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    p2, q2 = p + n * offset, q + n * offset
    for a, b in ((p, p2), (q, q2)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.5)
    ax.annotate("", xy=q2, xytext=p2, arrowprops=dict(arrowstyle="<->", color=color, lw=0.7, shrinkA=0, shrinkB=0))
    m = (p2 + q2) / 2 + n * 4
    ang = np.rad2deg(np.arctan2(d[1], d[0]))
    if ang > 90 or ang < -90:
        ang += 180
    ax.text(m[0], m[1], text, fontsize=fs, color=color, ha="center", va="center", rotation=ang,
            rotation_mode="anchor", bbox=dict(fc=SURFACE, ec="none", pad=0.5))


def board(ax, poly):
    ax.add_patch(Polygon(poly, closed=True, fc=WOOD, ec=INK2, lw=0.8, zorder=2))


def platen(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc=STONE, ec=INK2, lw=0.8, hatch="////", zorder=1))


fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))

# --- (a) closing ("compression"): the L lies like a "<", ram pushes the leg ends together
ax = axes[0]
leg1, leg2, e1, e2 = l_corner(45, (0, 0))          # leg 1 goes up-right, leg 2 down-right
board(ax, leg2)
board(ax, leg1)
top = leg1[:, 1].max()
bot = leg2[:, 1].min()
x_axis = leg1[1][0]        # load axis through the outer end corners of both legs (= 150 cos 45)
platen(ax, x_axis - 30, top + 2, 60, 14)
platen(ax, x_axis - 90, bot - 16, 180, 14)
ax.annotate("", xy=(x_axis, top + 16), xytext=(x_axis, top + 50), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.text(x_axis + 6, top + 36, "F  (8 mm/min)", fontsize=8, color=INK)
ax.annotate("", xy=(x_axis, bot - 16), xytext=(x_axis, bot - 50), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.text(x_axis + 6, bot - 40, "R", fontsize=8, color=INK)
ax.plot([x_axis, x_axis], [bot - 55, top + 55], color=MUTED, lw=0.6, ls="--", zorder=0)
dim(ax, leg1[0], leg2[3], "150 mm", -14)                       # outer face of the lower leg, from the corner
dim(ax, leg1[3] + e1 * T, leg1[2], "132 mm", 14)               # inner face of the upper leg, from the inner corner
p18 = np.array([0.0, 0.0]) + e1 * 80
dim(ax, p18, p18 + e2 * T, "", 0)
ax.text(*(p18 + e2 * (T + 6)), "18 mm", fontsize=7.5, color=INK2, ha="left", va="top", rotation=45, rotation_mode="anchor")
corner_c = np.array([0.0, 0.0]) + e2 * T + e1 * T              # inner corner = "c" of Fig. 4(b)
assert abs((x_axis - corner_c[0]) - 80.6) < 0.1                # the paper's lever arm, reproduced
ax.plot(*corner_c, "o", ms=4, color=INK, zorder=4)
ax.text(corner_c[0] + 3, corner_c[1] + 3, "c", fontsize=8, color=INK, ha="left", va="bottom")
ax.annotate("", xy=(x_axis, corner_c[1]), xytext=(corner_c[0], corner_c[1]),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2, shrinkA=0, shrinkB=0))
ax.text((corner_c[0] + x_axis) / 2 + 4, corner_c[1] - 7, "80.6 mm\nlever arm", fontsize=8, color=INK, ha="center", va="top")
ax.set_title("(a) Closing (\"compression\"): the corner\nis squeezed shut between the platens", fontsize=9.5)
ax.set_xlim(-45, 175)
ax.set_ylim(bot - 62, top + 62)
ax.set_aspect("equal", adjustable="box")
clean(ax, hide_axes=True)

# --- (b) opening ("tension"): the L stands as a "Λ" on sliding plates, ram pushes the apex down
ax = axes[1]
# The Λ: leg 1 along -45 deg (down-right), leg 2 along -135 deg (down-left)
a1 = np.deg2rad(-45)
e1 = np.array([np.cos(a1), np.sin(a1)])
e2 = np.array([-np.cos(np.pi / 4), -np.sin(np.pi / 4)])
apex = np.array([0.0, 0.0])
leg1 = np.array([apex, apex + e1 * LEG, apex + e1 * LEG + e2 * T, apex + e2 * T])
leg2 = np.array([apex + e2 * T, apex + e2 * T + e1 * T, apex + e2 * LEG + e1 * T, apex + e2 * LEG])
board(ax, leg2)
board(ax, leg1)
feet_y = min(leg1[:, 1].min(), leg2[:, 1].min())
assert abs(leg1[2][0] - 93.3) < 0.1 and abs(leg1[2][1] - feet_y) < 1e-9   # lowest foot corner sits on the roller
r = 5.0
for sx in (-1, 1):
    ax.add_patch(Circle((sx * 93.3, feet_y - r), r, fc=SURFACE, ec=INK2, lw=0.8, zorder=3))
platen(ax, -150, feet_y - 2 * r - 12, 300, 12)
ax.annotate("", xy=(0, 14), xytext=(0, 52), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.text(6, 36, "F  (8 mm/min)", fontsize=8, color=INK)
platen(ax, -22, 2, 44, 12)
for sx in (-1, 1):
    ax.annotate("", xy=(sx * 93.3, feet_y - 2 * r - 12), xytext=(sx * 93.3, feet_y - 2 * r - 46),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    ax.text(sx * 93.3 + 5, feet_y - 2 * r - 40, "R/2", fontsize=8, color=INK)
ax.plot([0, 0], [feet_y - 60, 60], color=MUTED, lw=0.6, ls="--", zorder=0)
dim(ax, leg1[0], leg1[1], "150 mm", 14)
dim(ax, leg2[0] + e1 * T, leg2[2], "132 mm", 14)
p18 = apex + e2 * 80
dim(ax, p18, p18 + e1 * T, "", 0)
ax.text(*(p18 - e1 * 6), "18 mm", fontsize=7.5, color=INK2, ha="right", va="center", rotation=-45, rotation_mode="anchor")
yd = feet_y - 2 * r - 14
dim(ax, (-93.3, yd), (0, yd), "93.3 mm", 0)
dim(ax, (0, yd), (93.3, yd), "93.3 mm", 0)
ax.text(0, feet_y - 2 * r - 50, "rollers let the feet splay outward", fontsize=7.5, color=INK2, ha="center", va="top",
        bbox=dict(fc=SURFACE, ec="none", pad=1))
ax.set_title("(b) Opening (\"tension\"): pushing the apex\ndown splays the legs on rollers", fontsize=9.5)
ax.set_xlim(-160, 160)
ax.set_ylim(feet_y - 78, 62)
ax.set_aspect("equal", adjustable="box")
clean(ax, hide_axes=True)
fig.suptitle("Test set-up: 18 mm particleboard L-corners (150 × 150 × 400 mm) bent closed and open",
             fontsize=10.5, fontweight="semibold", y=1.0)
fig.tight_layout()
save(fig, "janikova2025-rta-eccentric-joint-strength-1.png")  # set-up


# ============================================================================
# Figure 3: strength versus stiffness, the two rankings disagree
# ============================================================================
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name, K, M, col in [("closing", K_CLOSE, M_CLOSE, BLUE), ("opening", K_OPEN, M_OPEN, ORANGE)]:
    ax.plot([K[t] for t in TYPES], [M[t] for t in TYPES], "o", color=col, ms=7, mec="white", mew=0.8,
            label=name, zorder=3)
    for t in TYPES:
        dx, dy = 8, 0.6
        if name == "closing" and t == "A":
            dy = -2.8
        if name == "closing" and t == "E":
            dx, dy = -9, 1.2
        if name == "opening" and t in ("F", "E"):
            dy = -2.8 if t == "F" else 1.0
        if name == "opening" and t == "D":
            dy = 1.6
        ax.text(K[t] + dx, M[t] + dy, t, fontsize=8, color=INK2, ha="left" if dx > 0 else "right", va="center")
ax.set_xlabel("rotational stiffness (Nm/rad)")
ax.set_ylabel("peak bending moment $M_{max}$ (Nm)")
ax.set_title("Strength is not stiffness: G is the strongest but only\nmid-stiff; F is the stiffest but mid-strength",
             fontsize=10)
ax.set_xlim(0, 440)
ax.set_ylim(0, 52)
ax.legend(loc="upper left", fontsize=8, title="corner bent…", title_fontsize=8)
clean(ax)
save(fig, "janikova2025-rta-eccentric-joint-strength-3.png")
