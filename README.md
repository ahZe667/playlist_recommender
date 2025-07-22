Flask web application that helps you build a personalised Spotify‑style playlist
by rating a handful of sample tracks

Features:

Clean, responsive UI – vanilla Flask templates with inline CSS.

Duplicate‑aware pre‑processing – collapses identical track name + artist pairs before feature extraction.

Rich audio vectors – z‑scored numerical features plus circular encodings for key & time‑signature.

Adaptive centroid estimation – chooses the best K‑Means cluster count (2–6) using silhouette score.

Positive & negative feedback – pulls similar songs closer, pushes disliked ones away (NEG_PENALTY).

Diversity‑aware re‑ranking – greedily selects songs that are both relevant and cover different sub‑regions.

Popularity boost – mixes similarity with Spotify popularity (tunable POPULAR_W).

Algorithm Overview:

Deduplication – collapses rows sharing the same (track name, artist), keeps mean popularity.

Feature Vector Build – z‑scale numeric columns, encode key as sine/cosine, one‑hot time signature.

Rating UI – user rates or skips songs; ratings are stored in the session.

Centroid Estimation – finds 1 – 6 positive clusters with the highest silhouette score.

Similarity Calculation – combines cosine & Euclidean distances to positive centroids, subtracts a negative penalty.

Diversification – greedy selection that penalises intra‑playlist similarity.

Scoring – blends similarity with popularity, sorts descending, returns the top N tracks.
