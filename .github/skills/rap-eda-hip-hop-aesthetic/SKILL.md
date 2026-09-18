# Rap EDA Hip Hop Aesthetic Skill

**Purpose**: Define and apply a consistent hip hop-inspired visual aesthetic to all Rap corpus exploratory data analysis (EDA) visualizations.

**When to use**:
- Creating or updating rap corpus analysis visualizations
- Generating word clouds, histograms, temporal plots, or any matplotlib figures
- Maintaining cohesive visual identity across `R00_corpus_analysis.ipynb` and `R01_pos_analysis.ipynb`
- Extending visualizations at the repository root

## Color Palette

### Primary Hip Hop Aesthetic
```python
# Colors
BACKGROUND_COLOR = 'black'      # Dark urban aesthetic
COLORMAP_PRIMARY = 'YlOrRd'      # Yellow → Orange → Red gradient
COLORMAP_ALT = 'hot'             # Alternative: high-contrast hot palette

# Hex values for direct use
HIP_HOP_PALETTE = {
    'black': '#000000',
    'dark_gray': '#1a1a1a',
    'gold': '#FFD700',
    'orange': '#FF8C00',
    'red': '#DC143C',
    'light_gold': '#FFA500',
    'crimson': '#DC143C'
}
```

## Implementation Guidelines

### 1. Word Cloud Visualizations

For POS comparisons or any distribution with a very dominant head, use a
strong rank-weighted cloud. Keep raw frequencies in CSV exports, but map ranks
to a bounded visual score so leading terms dominate while the long tail stays
visible:

```python
top_words = word_freq.most_common(100)
rank_denominator = max(len(top_words) - 1, 1)
rank_scores = {
    word: 1.0 - 0.90 * (rank / rank_denominator) ** 0.65
    for rank, (word, _) in enumerate(top_words)
}
```

Use `relative_scaling=1.0`, `max_font_size=120`, and `min_font_size=10` with
these scores. The visual range is 10% to 100% of the maximum size. Bar charts
are intentionally omitted in the current POS notebook; preserve exact raw
frequencies in CSV exports instead.

For the sixth POS-cloud panel, combine `PUNCT` and `SYM` surface forms with
alphabetic `INTJ` lemmas. Label it `PUNCT / SYM / INTJ` and describe `INTJ` as
a reproducible proxy for interjections and likely onomatopoeia, not as an
exhaustive detector.

For distributions where preserving frequency magnitude is more important
than equal visibility, use the logarithmic variant below.

```python
from wordcloud import WordCloud
import numpy as np

# Apply logarithmic scaling to frequencies
top_200_words = dict(word_freq.most_common(200))
log_freq = {word: np.log1p(freq) for word, freq in top_200_words.items()}

# Generate word cloud with hip hop aesthetic
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color='black',           # Dark background
    colormap='YlOrRd',                  # Gold/Orange/Red gradient
    max_words=200,
    relative_scaling=0.7,
    min_font_size=10,
    prefer_horizontal=0.7
).generate_from_frequencies(log_freq)

# Display
fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Your Title', fontsize=16, fontweight='bold', pad=20)
plt.savefig('output.png', dpi=300, bbox_inches='tight', facecolor='black')
```

### 2. Matplotlib Figures (Histograms, Line Plots)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set hip hop aesthetic globally
plt.style.use('dark_background')
sns.set_palette("YlOrRd")

# Configure figure with dark background
fig, ax = plt.subplots(figsize=(14, 6), facecolor='#1a1a1a')
ax.set_facecolor('#000000')

# Plot with hip hop colors
ax.plot(x, y, color='#FFD700', linewidth=2.5, marker='o')  # Gold line
ax.fill_between(x, y, alpha=0.3, color='#FF8C00')          # Orange fill

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#FFD700')                     # Gold axes
ax.spines['bottom'].set_color('#FFD700')
ax.grid(axis='y', alpha=0.3, color='#FF8C00', linestyle='--')
ax.set_title('Your Title', fontsize=14, fontweight='bold', color='#FFD700')
ax.set_xlabel('X Label', color='#FFD700')
ax.set_ylabel('Y Label', color='#FFD700')
ax.tick_params(colors='#FFD700')

plt.tight_layout()
plt.savefig('output.png', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
```

### 3. Dual-Axis Plots (Year Evolution)
```python
fig, ax1 = plt.subplots(figsize=(14, 6), facecolor='#1a1a1a')
ax1.set_facecolor('#000000')

# First axis: Gold/Yellow
ax1.plot(years, values1, 'o-', color='#FFD700', linewidth=2.5, markersize=6, label='Metric 1')
ax1.set_ylabel('Metric 1', color='#FFD700', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#FFD700')
ax1.grid(axis='y', alpha=0.2, color='#FFD700')

# Second axis: Orange/Red
ax2 = ax1.twinx()
ax2.plot(years, values2, 's-', color='#FF8C00', linewidth=2.5, markersize=6, label='Metric 2')
ax2.set_ylabel('Metric 2', color='#FF8C00', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#FF8C00')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
          facecolor='#1a1a1a', edgecolor='#FFD700', framealpha=0.95)

ax1.spines['left'].set_color('#FFD700')
ax1.spines['bottom'].set_color('#FFD700')
ax2.spines['right'].set_color('#FF8C00')
ax1.spines['top'].set_visible(False)

ax1.set_xlabel('X Label', color='#FFD700', fontweight='bold')
plt.tight_layout()
plt.savefig('output.png', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
```

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `background_color` | `'black'` | Urban, high-contrast aesthetic |
| `colormap` | `'YlOrRd'` | Gold → Orange → Red (hip hop/rap vibes) |
| `facecolor` | `'#1a1a1a'` (dark gray) | Slightly lighter than pure black for readability |
| `figure_dpi` | `300` | High-resolution album cover quality |
| `relative_scaling` (wordcloud) | `0.7` | Balances word size distribution |
| `log_scaling` | `np.log1p(freq)` | Makes smaller words visible (logarithmic frequency) |
| `prefer_horizontal` | `0.7` | 70% horizontal text orientation |

## Palettes by Use Case

### High Contrast (Word Clouds)
```python
colormap='YlOrRd'  # Best for dark backgrounds
```

### Temporal Data (Time Series)
```python
# Gold for primary metric, Orange/Red for secondary
primary_color = '#FFD700'   # Gold
secondary_color = '#FF8C00' # Orange
accent_color = '#DC143C'    # Crimson
```

### Categorical/Frequency Data (Histograms)
```python
# Use YlOrRd colormap with black background
# Add mean/median lines in gold
ax.axvline(mean, color='#FFD700', linestyle='--', linewidth=2, label=f'Mean: {mean:.0f}')
ax.axvline(median, color='#FFA500', linestyle='-.', linewidth=2, label=f'Median: {median:.0f}')
```

## File Naming Convention
All outputs should follow this pattern:
```
images/NN_descriptive_name_scope.png
export/descriptive_name_scope.csv
```

Where NN is a sequence number (01, 02, 03, etc.)

## Validation Checklist

Before considering a visualization "complete":
- [ ] Background is black (`#000000`) or dark gray (`#1a1a1a`)
- [ ] Color palette uses YlOrRd or gold/orange/red gradient
- [ ] Text/axes are gold (`#FFD700`) for visibility on dark background
- [ ] DPI is 300 for print/publication quality
- [ ] Figure saved with `facecolor` parameter to maintain dark theme
- [ ] All labels are in English (unless French is required for specific context)
- [ ] Title and axis labels are bold and clearly visible
- [ ] Legend (if present) has dark background with light borders
- [ ] Logarithmic scaling applied to frequency distributions where appropriate

## Example: Complete Hip Hop EDA Workflow

```python
# 1. Setup
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.style.use('dark_background')
images_dir = Path('images')

# 2. Word Cloud
from wordcloud import WordCloud
log_freq = {w: np.log1p(f) for w, f in word_freq.most_common(200)}
wordcloud = WordCloud(
    width=1600, height=800,
    background_color='black',
    colormap='YlOrRd',
    max_words=200, relative_scaling=0.7
).generate_from_frequencies(log_freq)

fig, ax = plt.subplots(figsize=(16, 8), facecolor='#1a1a1a')
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
plt.savefig(images_dir / 'word_cloud.png', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')

# 3. Histogram with Hip Hop Aesthetic
fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a1a')
ax.set_facecolor('#000000')
ax.hist(data, bins=50, color='#FF8C00', edgecolor='#FFD700', alpha=0.7)
ax.axvline(np.mean(data), color='#FFD700', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.0f}')
ax.set_xlabel('Label', color='#FFD700', fontweight='bold')
ax.set_ylabel('Frequency', color='#FFD700', fontweight='bold')
ax.tick_params(colors='#FFD700')
ax.spines['left'].set_color('#FFD700')
ax.spines['bottom'].set_color('#FFD700')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(images_dir / 'histogram.png', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')

plt.show()
```

## Notes

- **Why log scaling?** Frequencies follow a Zipfian distribution where a few words (comme, vie, rien) dominate. Log scaling using `np.log1p(freq)` compresses the distribution, making smaller frequencies visible without losing the ranking order.

- **Why black background?** Evokes urban/street culture aesthetic central to hip hop, provides maximum contrast for gold/orange text, and reduces eye strain for dark mode users.

- **Why YlOrRd colormap?** Yellow (gold) = prestige/wealth, Orange = energy/vitality, Red = passion/strength. This gradient matches hip hop cultural symbolism and album art conventions.

---

**Last Updated**: 2026-09-15
**Project**: Rap Corpus EDA Analysis
**Status**: Active
