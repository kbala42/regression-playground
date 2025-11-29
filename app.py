import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Streamlit temel ayar
# -----------------------------
st.set_page_config(page_title="Regresyon Playground", page_icon="📉")

st.title("📉 Regresyon Playground – En Uygun Doğruyu Bul")
st.write(
    """
Bu laboratuvarda rastgele üretilmiş nokta bulutu üzerinde,
**en iyi uyum sağlayan doğruyu** keşfetmeye çalışacaksın.

- Nokta sayısını ve gürültü seviyesini seç  
- Eğim (**m**) ve başlangıç değeri (**b**) için tahmin yap  
- Hatanın (MSE) nasıl değiştiğini gözlemle  
- İstersen, en küçük kareler yönteminin bulduğu 'en iyi' doğruyu da gör
"""
)

st.markdown("---")


# -----------------------------
# Veri üretimi ayarları
# -----------------------------
st.subheader("1️⃣ Veri Setini Oluştur")

col_data1, col_data2 = st.columns(2)

with col_data1:
    n_points = st.slider(
        "Nokta sayısı",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

with col_data2:
    noise_level = st.slider(
        "Gürültü seviyesi",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Gürültü arttıkça noktalar doğrunun etrafında daha dağınık olur.",
    )

st.markdown("**Gerçek (gizli) doğruyu belirle:**")

col_true1, col_true2 = st.columns(2)
with col_true1:
    true_m = st.slider(
        "Gerçek eğim (m_true)",
        min_value=-3.0,
        max_value=3.0,
        value=1.0,
        step=0.5,
    )
with col_true2:
    true_b = st.slider(
        "Gerçek başlangıç değeri (b_true)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
    )

seed = st.number_input(
    "Rastgelelik için seed (isteğe bağlı, aynı sayıyı girersen aynı veri oluşur)",
    min_value=0,
    max_value=10_000,
    value=0,
    step=1,
)

# -----------------------------
# Veriyi üret
# -----------------------------
rng = np.random.default_rng(seed)
x = np.linspace(0, 10, n_points)
y_true_line = true_m * x + true_b
y_obs = y_true_line + noise_level * rng.standard_normal(size=n_points)


# -----------------------------
# Öğrencinin tahmin ettiği doğru
# -----------------------------
st.markdown("---")
st.subheader("2️⃣ Kendi Doğrunu Tahmin Et")

col_guess1, col_guess2 = st.columns(2)
with col_guess1:
    guess_m = st.slider(
        "Tahmin ettiğin eğim (m)",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
    )
with col_guess2:
    guess_b = st.slider(
        "Tahmin ettiğin başlangıç değeri (b)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
    )

y_pred_guess = guess_m * x + guess_b

# Ortalama kare hata (Mean Squared Error)
mse_guess = float(np.mean((y_obs - y_pred_guess) ** 2))

st.write(f"Seçtiğin doğrunun **MSE (ortalama kare hata)** değeri: **{mse_guess:.3f}**")

if mse_guess < 1.0:
    st.caption("Harika! Hata oldukça küçük, doğru çizgin noktalarla iyi uyuşuyor.")
elif mse_guess < 5.0:
    st.caption("Fena değil. Biraz daha m ve b ile oynayıp hatayı düşürmeyi deneyebilirsin.")
else:
    st.caption("Hata büyük görünüyor. Muhtemelen eğim veya başlangıç değeri hedef doğrudan uzak.")


# -----------------------------
# En küçük kareler ile 'en iyi' doğru
# -----------------------------
st.markdown("---")
st.subheader("3️⃣ En Küçük Kareler (Least Squares) ile 'En İyi' Doğru")

show_best = st.checkbox(
    "En küçük kareler yönteminin bulduğu 'en iyi' doğruyu da göster",
    value=True,
)

if show_best:
    # X matrisi: [x, 1]
    X = np.vstack([x, np.ones_like(x)]).T
    best_m, best_b = np.linalg.lstsq(X, y_obs, rcond=None)[0]
    y_best = best_m * x + best_b
    mse_best = float(np.mean((y_obs - y_best) ** 2))

    st.write(
        f"En küçük kareler ile bulunan doğru: "
        f"**y = {best_m:.2f} · x + {best_b:.2f}**"
    )
    st.write(f"Bu doğrunun MSE değeri: **{mse_best:.3f}**")
else:
    y_best = None


# -----------------------------
# Görselleştirme
# -----------------------------
st.markdown("---")
st.subheader("4️⃣ Grafikte İncele")

fig, ax = plt.subplots(figsize=(7, 5))

# Nokta bulutu (gözlenen veriler)
ax.scatter(x, y_obs, label="Veri noktaları")

# Gerçek doğru (gizli model)
ax.plot(x, y_true_line, linestyle="--", label="Gerçek doğru (gizli)")

# Öğrencinin tahmini
ax.plot(x, y_pred_guess, linestyle="-", label="Senin doğrun")

# En iyi doğru (least squares)
if show_best and y_best is not None:
    ax.plot(x, y_best, linestyle=":", label="En küçük kareler doğrusu")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Regresyon Playground")
ax.legend()
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

st.pyplot(fig)


# -----------------------------
# Açıklama / Öğretmen kutusu
# -----------------------------
st.markdown("---")
st.info(
    "Bu görselleştirme, regresyon kavramını sezgisel olarak tanıtmak için tasarlanmıştır. "
    "Her veri noktası ile çizdiğin doğru arasındaki dikey farklar (hatalar) karesinin ortalaması, "
    "MSE değeri olarak hesaplanır. MSE ne kadar küçükse, doğru o kadar iyi uyum sağlar."
)

with st.expander("👩‍🏫 Öğretmen Kutusu – En Küçük Kareler Fikri"):
    st.write(
        r"""
**Amaç:** Verilen $(x_i, y_i)$ noktalarına en iyi uyan

\\[
y = m x + b
\\]

doğrusunu bulmak.

En küçük kareler yöntemi, **tüm noktalar için hata karelerinin toplamını** en küçük yapan
$(m, b)$ ikilisini seçer:

\\[
\text{MSE}(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (m x_i + b))^2
\\]

Bu labda öğrenciler:

- Önce `m` ve `b` değerlerini **deneyerek** MSE'yi küçültmeye çalışır,  
- Sonra en küçük karelerin bulduğu 'en iyi' doğruyu görerek,  
  denemeleriyle matematiksel çözüm arasındaki farkı/sezgiyi karşılaştırırlar.
"""
    )

st.caption(
    "Bu modül, lise düzeyinde regresyon ve hata kavramına görsel bir giriş sağlamak için tasarlanmıştır."
)
