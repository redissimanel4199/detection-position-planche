# detection-position-planche
Détection de position de planche en direct avec Python, OpenCV et MediaPipe

# Détection de position de planche (plank)

Projet personnel : détection de la position de planche en direct sur caméra,
avec chronomètre de maintien. Initié pendant le Master 2 MIAGE (Université Paris-Saclay)
à partir de photos, puis poursuivi et amélioré en autonomie.

## Fonctionnement
1. OpenCV capture les images de la webcam.
2. MediaPipe Pose détecte les points du corps.
3. Une planche est reconnue par des règles géométriques (angle épaule-hanche-cheville
   proche de 180°, alignement des hanches, position des coudes).
4. Un chronomètre mesure la durée, avec une tolérance de 0,7 s pour éviter les coupures
   dues aux clignotements de la détection.

## Installation
pip install -r requirements.txt

## Utilisation
- Notebook : ouvrir `detection_planche.ipynb` et exécuter la cellule principale.
- Script : `python plank_timer.py --camera 0` (touche q pour quitter).

Se placer de profil, en pied, à environ 2 m de la caméra.

## Limites connues
- Fonctionne de profil ; les vues de face ou de dos ne sont pas gérées.
- Les seuils dépendent de la caméra et de l'éclairage.
- Pas encore de mesure de précision sur un jeu de vidéos annotées.

## À faire
- [ ] Application web Python (Flask) : affichage en direct et historique.
- [ ] Mesurer la précision et les FPS.
- [ ] Détecter les erreurs de posture.
