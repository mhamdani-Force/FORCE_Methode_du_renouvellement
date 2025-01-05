import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import math
from scipy.stats import expon, poisson

# -----------------------------------------------------------------------------
# SCRIPT CORRIGÉ POUR ÉVITER LES DEUX AVERTISSEMENTS DE PANDAS
# -----------------------------------------------------------------------------
# 1) On passe l'argument include_groups=False dans groupby.apply
# 2) On remplace resample('A') par resample('YE')
# -----------------------------------------------------------------------------

# Entrée utilisateur pour le seuil
x0 = float(input("Entrez le seuil (x0) : "))

# Sélection du fichier Excel
Tk().withdraw()  # Masquer la fenêtre principale
file_path = askopenfilename(filetypes=[("Fichiers Excel", "*.xlsx;*.xls")])

# -----------------------------------------------------------------------------
# 1. LECTURE ET PRÉPARATION DES DONNÉES
# -----------------------------------------------------------------------------

data = pd.read_excel(file_path)

# Conversion si nécessaire
if not pd.api.types.is_datetime64_any_dtype(data['Date']):
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')

if not pd.api.types.is_string_dtype(data['H']):
    data['H'] = data['H'].astype(str)

data['Datetime'] = pd.to_datetime(
    data['Date'].astype(str) + ' ' + data['H'],
    format='%Y-%m-%d %H:%M'
)

data['Débit'] = pd.to_numeric(
    data['Débit'].astype(str).str.replace(',', '.'),
    errors='coerce'
)

full_range = pd.date_range(
    start=data['Datetime'].min(),
    end=data['Datetime'].max(),
    freq='15min'
)

all_data = pd.DataFrame({'Datetime': full_range})
all_data = pd.merge(all_data, data[['Datetime', 'Débit']], 
                    on='Datetime', how='left')

all_data['Manquantes'] = all_data['Débit'].isna()

# -----------------------------------------------------------------------------
# 2. IDENTIFICATION DES PICS INDÉPENDANTS
# -----------------------------------------------------------------------------

all_data['Au_dessus'] = all_data['Débit'] > x0

event_id = 0
event_ids = []
currently_above = False

for above in all_data['Au_dessus']:
    if above:
        if not currently_above:
            event_id += 1
            currently_above = True
        event_ids.append(event_id)
    else:
        currently_above = False
        event_ids.append(None)

all_data['event_id'] = event_ids

# On sélectionne uniquement les colonnes utiles pour le groupby
df_events = all_data.loc[all_data['event_id'].notna(), 
                         ['Datetime', 'Débit', 'event_id']].copy()

grouped = df_events.groupby('event_id', group_keys=False)

# include_groups=False évite le DeprecationWarning de pandas
peaks = grouped.apply(
    lambda g: g.loc[g['Débit'].idxmax()],
    include_groups=False
).reset_index(drop=True)

peaks['Excès'] = peaks['Débit'] - x0
n = len(peaks)

duree_jours = (all_data['Datetime'].max() - all_data['Datetime'].min()).days
N = duree_jours / 365.25

if N == 0 or n == 0:
    raise ValueError("Erreur : N ou n = 0, vérifiez vos données.")

lambda_rate = n / N
mean_exceedance = peaks['Excès'].mean()
k = lambda_rate

# -----------------------------------------------------------------------------
# 3. LOIS DE POISSON ET EXPONENTIELLE
# -----------------------------------------------------------------------------

# -------------------------------
# 3.1 Loi de Poisson
# -------------------------------
# On met 'Datetime' en index pour resampler
peaks.set_index('Datetime', inplace=True)
# Remplacer 'A' (déprécié) par 'YE' (year-end)
poisson_counts = peaks.resample('YE').size()
poisson_mean = poisson_counts.mean()

plt.figure(figsize=(12, 5))
bins_range = range(0, int(poisson_counts.max()) + 2)
plt.hist(poisson_counts, bins=bins_range, density=True,
         alpha=0.6, color='skyblue', label='Observé')

k_vals = np.arange(0, int(poisson_counts.max()) + 2)
plt.plot(k_vals, poisson.pmf(k_vals, poisson_mean), 'ro-',
         label='Poisson théorique')

plt.xlabel('Nombre de dépassements par an')
plt.ylabel('Fréquence (densité)')
plt.title('Vérification de la loi de Poisson')
plt.legend()
plt.grid(True)
plt.show()

# On remet un index par défaut (pour pouvoir tracer ensuite peaks['Datetime'])
peaks.reset_index(inplace=True)

# -------------------------------
# 3.2 Loi exponentielle
# -------------------------------
plt.figure(figsize=(12, 5))
plt.hist(peaks['Excès'], bins=30, density=True,
         alpha=0.6, color='green', label='Observé')
x_sorted = np.sort(peaks['Excès'])
plt.plot(x_sorted, expon.pdf(x_sorted, scale=mean_exceedance), 'r-',
         label='Exponentielle théorique')
plt.xlabel('Excès (x - x0)')
plt.ylabel('Densité')
plt.title('Vérification de la loi exponentielle')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# 4. CALCUL DES QUANTILES
# -----------------------------------------------------------------------------

T_list = [5, 10, 25, 50, 100]
quantiles = []

for T in T_list:
    # xT = x0 + beta * ln(T / k)
    ratio = (T / k) if (T / k) > 1e-5 else 1e-5
    xT = x0 + mean_exceedance * np.log(ratio)
    quantiles.append(xT)

# -----------------------------------------------------------------------------
# 5. AFFICHAGE ET ENREGISTREMENT
# -----------------------------------------------------------------------------

parametres = pd.DataFrame({
    'Paramètre': [
        'Seuil (x0)',
        'Nombre de dépassements (n)',
        'Lambda (taux/an)',
        'Moyenne des excès (beta)',
        'Facteur k (= n/N)'
    ],
    'Valeur': [
        x0,
        n,
        lambda_rate,
        mean_exceedance,
        k
    ]
})

print("Paramètres de calcul :")
print(parametres)
parametres.to_csv('parametres_calcul.csv', index=False)

resultats = pd.DataFrame({
    'Période de retour (ans)': T_list,
    'Quantile (m³/s)': quantiles
})
print("\nRésultats des quantiles :")
print(resultats)
resultats.to_csv('resultats_quantiles.csv', index=False)
print("Résultats enregistrés dans 'resultats_quantiles.csv'")

plt.figure(figsize=(12, 6))
plt.plot(all_data['Datetime'], all_data['Débit'], 
         label='Débit observé', color='blue')
plt.axhline(y=x0, color='r', linestyle='--', label='Seuil')

plt.scatter(peaks['Datetime'], peaks['Débit'],
            color='red', label='Pics retenus')

plt.fill_between(all_data['Datetime'],
                 0, all_data['Débit'].max(),
                 where=all_data['Manquantes'], color='yellow', alpha=0.4,
                 label='Données manquantes')

plt.xlabel('Date')
plt.ylabel('Débit (m³/s)')
plt.title('Analyse des dépassements - Méthode du renouvellement ')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
