import matplotlib.pyplot as plt
import numpy as np
import os

models = ['YuE', 'StableAudio', 'JASCO', 'MAGNeT', 'AudioX']
genres = ['Pop', 'Rock', 'Country', 'Jazz', 'Classical']
model_colors = {
    "YuE": "#1f77b4", "StableAudio": "#ff7f0e", "JASCO": "#2ca02c",
    "MAGNeT": "#d62728", "AudioX": "#9467bd"
}

data = {
    'Pop': {
        'YuE': [('Electronic', 0.101285), ('Pop', 0.043145), ('R&B', 0.025846), ('Disco', 0.017872), ('Hip hop', 0.014318)],
        'StableAudio': [('Electronic', 0.1780616), ('Hip hop', 0.0394124), ('R&B', 0.0262816), ('Disco', 0.0181772), ('Pop', 0.0171424)],
        'JASCO': [('Electronic', 0.087825), ('New-age', 0.02693), ('R&B', 0.008756), ('Country', 0.007863), ('Soul', 0.006358)],
        'MAGNeT': [('Electronic', 0.22305), ('Disco', 0.01665), ('Pop', 0.00878), ('New-age', 0.00796), ('Hip hop', 0.0072)],
        'AudioX': [('Electronic', 0.14352), ('New-age', 0.01126), ('Pop', 0.0074), ('Disco', 0.00721), ('R&B', 0.00666)]
    },
    'Rock': {
        'YuE': [('Rock', 0.046919), ('Pop', 0.016419), ('R&B', 0.014152), ('Electronic', 0.010236), ('Funk', 0.007667)],
        'StableAudio': [('Rock', 0.0481856), ('Blues', 0.0121624), ('Electronic', 0.0062792), ('Pop', 0.0058606), ('Country', 0.0054468)],
        'JASCO': [('Rock', 0.05994), ('Electronic', 0.0168), ('Blues', 0.0082), ('Pop', 0.00806), ('R&B', 0.00785)],
        'MAGNeT': [('Electronic', 0.04584), ('Rock', 0.0421), ('R&B', 0.00624), ('Pop', 0.0061), ('Blues', 0.00558)],
        'AudioX': [('Rock', 0.02028), ('Electronic', 0.01123), ('Blues', 0.00432), ('Country', 0.00356), ('Classical', 0.00212)]
    },
    'Country': {
        'YuE': [('Country', 0.016495), ('Blues', 0.009845), ('New-age', 0.008601), ('Electronic', 0.006882), ('Rock', 0.005605)],
        'StableAudio': [('Country', 0.025213), ('Rock', 0.0107048), ('Blues', 0.0106646), ('Electronic', 0.00619), ('New-age', 0.006125)],
        'JASCO': [('New-age', 0.01733), ('Electronic', 0.01437), ('Country', 0.00839), ('Blues', 0.00732), ('R&B', 0.00527)],
        'MAGNeT': [('Electronic', 0.03255), ('New-age', 0.02545), ('Country', 0.02399), ('Rock', 0.01617), ('Blues', 0.01126)],
        'AudioX': [('Country', 0.00955), ('Blues', 0.00691), ('Classical', 0.00665), ('New-age', 0.00521), ('Rock', 0.00458)]
    },
    'Jazz': {
        'YuE': [('Electronic', 0.045371), ('R&B', 0.013981), ('Soul', 0.011631), ('New-age', 0.011429), ('Pop', 0.010018)],
        'StableAudio': [('Jazz', 0.0649362), ('Blues', 0.0117722), ('Electronic', 0.0102228), ('Classical', 0.009182), ('New-age', 0.0079094)],
        'JASCO': [('Electronic ', 0.03578), ('Jazz', 0.01371), ('Blues', 0.01112), ('Reggae', 0.00942), ('Rock', 0.0075)],
        'MAGNeT': [('Electronic ', 0.03033), ('Hip hop', 0.02043), ('Jazz', 0.01927), ('R&B', 0.00963), ('Blues', 0.00873)],
        'AudioX': [('Electronic ', 0.03753), ('Jazz', 0.01572), ('Classical', 0.00319), ('New-age', 0.00293), ('Hip hop', 0.00283)]
    },
    'Classical': {
        'YuE': [('Electronic', 0.027522), ('R&B', 0.010567), ('New-age', 0.009852), ('Pop', 0.008874), ('Hip hop', 0.005138)],
        'StableAudio': [('New-age', 0.080739), ('Electronic', 0.015675), ('Classical', 0.007795), ('Country', 0.003143), ('Soul', 0.003102)],
        'JASCO': [('Classical', 0.045182), ('New-age', 0.027457), ('Electronic', 0.010575), ('Jazz', 0.002568), ('Soul', 0.001591)],
        'MAGNeT': [('Classical', 0.030052), ('New-age', 0.026317), ('Electronic', 0.01787), ('Jazz', 0.003743), ('Folk', 0.003155)],
        'AudioX': [('New-age', 0.029303), ('Classical', 0.020569), ('Electronic', 0.011572), ('Blues', 0.002088), ('Soul', 0.002044)]
    }
}

output_dir = os.path.join(os.path.dirname(__file__), "genre_pngs")
os.makedirs(output_dir, exist_ok=True)

saved_files = []
for genre in genres:
    fig, ax = plt.subplots(figsize=(14, 6))
    model_data = data[genre]
    bar_width = 0.18
    x = np.arange(5) * (len(models) + 1) * bar_width * 0.9

    for i, model in enumerate(models):
        predictions = model_data[model]
        labels, scores = zip(*predictions)
        x_positions = x + i * bar_width
        bars = ax.bar(x_positions, scores, width=bar_width, label=model, color=model_colors[model])

        for bar, label, score in zip(bars, labels, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{label}\n{score:.3f}", ha='center', va='bottom', fontsize=8)

    ax.set_title(f"{genre} - Top 5 Genre Predictions per Model")
    ax.set_ylabel("Confidence Score")
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels([f"#{i+1}" for i in range(5)])
    ax.margins(x=0.01)
    ax.set_ylim(0, 0.3)
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()

    file_path = os.path.join(output_dir, f"{genre}_genre_predictions.png")
    fig.savefig(file_path, dpi=300)
    saved_files.append(file_path)
    plt.close(fig)

saved_files