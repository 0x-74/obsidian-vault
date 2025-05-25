[paper link](https://github.com/TaoRuijie/TalkNet-ASD/tree/main?tab=readme-ov-file)
keywords to study
cross attention self attention
key terms
speaker diarization : splitting audio into continous segments based on who is speaking currently , making new segments each time speaker changes, without knowing speaker identity
most papers dont take into account temporal flow of audio and visual flows AND interaction between audio and visual signals
current papers use small audio segments from 200 to 600 ms but each humans consider entire sentence, 5 seconds has 15 words 200 ms cant even get 1 word
VAD : detecting speech amidst all acoustic audios, distant mics might be unintelligible and overlapping noise is another edge case
ASD takes motivation from using RNN or GRU on concatenated
Rahul Sharma, Krishna Somandepalli, and Shrikanth Narayanan. 2020. Crossmodal learning for audio-visual speech event localization. arXiv preprint arXiv:2003.04358 (2020).
[6] Ido Ariav and Israel Cohen. 2019. An end-to-end multimodal voice activity detection using wavenet encoder and residual networks. IEEE Journal of Selected Topics in Signal Processing 13, 2 (2019), 265–274.
[50] Yuan-Hang Zhang, Jingyun Xiao, Shuang Yang, and Shiguang Shan. 2019. MultiTask Learning for Audio-Visual Active Speaker Detection. (2019)