"""Sections retained from the team's synopsis (ML.pdf), with PDF-extraction artifacts fixed
(black-box, real-world, hour-of-day, transformer-based, grid-aware, decision-making, and classical,
high-performing, density-based, multi-horizon). In-text citations [n] were added."""

TITLE = ("Predictive Spatiotemporal Demand Modeling for Optimal Electric Vehicle "
         "Charging Station Placement Using Classical Machine Learning")

AUTHORS = [
    ("Prathamesh Patil", "Roll No.: 9", "Registration No.: 240962052"),
    ("Srijan Bera", "Roll No.: 17", "Registration No.: 240962300"),
    ("Snehal Tyagi", "Roll No.: 24", "Registration No.: 240962368"),
]

INDEX_TERMS = ("Electric vehicle charging demand, Poisson regression, negative binomial regression, "
               "spatial clustering, DBSCAN, queueing simulation, infrastructure planning, demand forecasting")

INTRODUCTION = [
    "The rapid adoption of electric vehicles (EVs) is straining existing charging infrastructure, which has largely "
    "grown in an unplanned, reactive manner rather than in response to actual usage demand [2]. Charging stations are "
    "frequently sited based on land availability or investor convenience rather than data-driven demand estimates, "
    "resulting in some areas being oversaturated with chargers while others remain underserved. This mismatch causes "
    "long wait times, underutilized assets, and slower EV adoption in high-need areas.",
    "This project addresses the problem by building a demand prediction and placement recommendation system using "
    "real-world charging session data. Historical session logs are aggregated into time-based demand features and "
    "modeled using count-based regression techniques suited to discrete event data, since charging sessions are "
    "fundamentally count occurrences rather than continuous quantities. Spatial clustering is then applied to the "
    "predicted demand surface to identify underserved high-demand zones, which are proposed as candidate locations "
    "for new stations.",
    "Unlike existing black-box forecasting approaches, this project emphasizes interpretable, classical statistical "
    "and machine learning methods, making the results directly explainable to urban planners and utility operators "
    "making real infrastructure decisions.",
]

LITERATURE = [
    "Research on EV charging demand prediction has grown substantially since 2020, driven by rising global EV adoption. "
    "A large share of recent work applies deep learning architectures—LSTM, GRU, Transformer, and graph "
    "convolutional networks—to forecast charging load at the station or city level [7], [8], [11], [12], using "
    "machine learning methods including RNN, LSTM, Bi-LSTM, GRU, CNN, and transformers on charging events from public "
    "stations. Similarly, transformer-based models have been proposed specifically for multi-horizon EV demand "
    "forecasting across short- and long-term windows, predicting charging demand at 7-day, 30-day, and 90-day "
    "horizons [1]. While these deep models often report strong accuracy, they require large training datasets and "
    "offer limited interpretability for infrastructure planners.",
    "A parallel body of work has moved toward data-driven infrastructure planning, not just forecasting. One study "
    "analyzing spatiotemporal charging patterns across Chinese megacities found that supply-demand imbalances vary "
    "sharply by both location and time, motivating precisely localized rather than uniform infrastructure strategies. "
    "Another framework develops a demand estimation model using spatiotemporal charging data and points-of-interest "
    "features as part of a broader grid-aware optimization pipeline [10], though it notes that many existing "
    "approaches focus solely on spatial or demand-side factors, often assuming grid upgrades are trivially "
    "achievable.",
    "For station siting specifically, GIS- and clustering-based methods dominate the literature. One approach applied "
    "DBSCAN clustering to assess charging access disparities between urban and suburban areas, while multi-criteria "
    "decision-making methods such as AHP, TOPSIS, and PROMETHEE have been widely used to rank candidate sites against "
    "criteria like traffic volume, population density, and land cost, with clustering techniques specifically used to "
    "group areas of high EV demand to guide placement decisions.",
    "Household- and session-level forecasting studies provide complementary insight into demand granularity. Ensemble "
    "and classical regression approaches have been shown to perform competitively for session- and energy-level "
    "forecasting without resorting to deep architectures, with custom weighted ensemble machine learning applied to "
    "session and energy forecasting at charging stations [3]–[5]. Broader systematic reviews of the field "
    "confirm that EV energy demand prediction techniques remain an active and rapidly evolving research area, but "
    "critical gaps persist around model interpretability and integration with real infrastructure decisions [2].",
]

GAPS = [
    ("Interpretability–accuracy tradeoff:",
     "Most high-performing models (CNN, Transformer, GCN-based) are black-box in nature, making them difficult for "
     "utility planners and municipal bodies to trust or justify decisions from."),
    ("Disconnect between forecasting and placement:",
     "Demand forecasting and station siting are largely studied as separate problems—forecasting papers rarely "
     "propose actionable placement recommendations, and siting papers (GIS/MCDM-based) often rely on static criteria "
     "rather than dynamically predicted demand."),
    ("Undercounting via continuous-variable models:",
     "Many studies treat charging session counts as continuous variables and apply standard linear/deep regression, "
     "ignoring that session counts are inherently discrete count data, for which Poisson/Negative Binomial regression "
     "is statistically more appropriate."),
    ("Limited use of density-based spatial clustering for hotspot identification:",
     "While k-means is common, density-based methods (DBSCAN) that naturally handle irregular-shaped demand hotspots "
     "and noise are comparatively underused in this specific pipeline (forecast → cluster → recommend)."),
]
GAPS_CLOSING = ("This project addresses these gaps by combining statistically appropriate count-based demand "
                "forecasting with density-based spatial clustering in a single interpretable, end-to-end pipeline "
                "that outputs actionable station-placement recommendations.")

OBJECTIVES = [
    "To collect and preprocess real-world EV charging session data and engineer temporal demand features "
    "(hour-of-day, day-of-week, holiday indicators, rolling averages).",
    "To develop a count-based regression model (Poisson / Negative Binomial regression) to predict station-level and "
    "zone-level charging demand.",
    "To apply density-based spatial clustering (DBSCAN) on predicted demand to identify underserved high-demand zones "
    "(“charging deserts”) relative to existing station density.",
    "To generate and visualize optimal candidate locations for new charging stations based on demand-coverage gap "
    "analysis.",
]

REFERENCES = [
    'S. Koohfar, W. Woldemariam, and A. Kumar, “Prediction of Electric Vehicles Charging Demand: A Transformer-Based Deep Learning Approach,” *Sustainability*, vol. 15, no. 3, p. 2105, 2023.',
    'F. Marzbani, A. H. Osman, and M. S. Hassan, “Electric Vehicle Energy Demand Prediction Techniques: An In-Depth and Critical Systematic Review,” *IEEE Access*, vol. 11, pp. 96242–96259, 2023.',
    'K. Akshay, G. H. Grace, K. Gunasekaran, and R. Samikannu, “Power Consumption Prediction for Electric Vehicle Charging Stations and Forecasting Income,” *Scientific Reports*, vol. 14, no. 1, p. 6497, 2024.',
    'R. N. Akshayaa, S. Sajidha, and R. Radha, “Session and Energy Forecasting for Electric Vehicle Charging Station Using Custom Weighted Ensemble Machine Learning,” *Results in Engineering*, vol. 26, p. 105104, 2025.',
    'F. Bellizio, B. Dijkstra, A. Fertig, J. Van Dijk, and P. Heer, “Machine Learning Approaches for the Prediction of Public EV Charge Point Flexibility,” *Sustainable Energy, Grids and Networks*, vol. 42, p. 101657, 2025.',
    'B. Al-Hanahi, I. Ahmad, D. Habibi, P. Pradhan, and M. A. Masoum, “A Charging Strategy for Large Commercial Electric Vehicle Fleets,” *IEEE Access*, vol. 12, pp. 46042–46058, 2024.',
    'L. Douaidi, S. M. Senouci, I. El Korbi, F. Harrou, and A. Yazici, “Federated Deep Learning for Enhanced Prediction of Electric Vehicle Charging Station Availability,” *Cluster Computing*, vol. 28, no. 6, p. 364, 2025.',
    'K. Hallak and O. Kem, “Adaptive Federated Learning Framework for Predicting EV Charging Stations Occupancy,” *International Journal of Transportation Science and Technology*, 2025.',
    'R. Lyu, H. Guo, K. Zheng, M. Sun, and Q. Chen, “Co-Optimizing Bidding and Power Allocation of an EV Aggregator Providing Real-Time Frequency Regulation Service,” *IEEE Transactions on Smart Grid*, vol. 14, no. 6, pp. 4594–4606, 2023.',
    'C. Li, Z. Dong, G. Chen, B. Zhou, J. Zhang, and X. Yu, “Data-Driven Planning of Electric Vehicle Charging Infrastructure: A Case Study of Sydney, Australia,” *IEEE Transactions on Smart Grid*, vol. 12, no. 4, pp. 3289–3304, 2021.',
    'F. Cheng and H. Liu, “Multi-Step Electric Vehicles Charging Loads Forecasting: An Autoformer Variant with Feature Extraction, Frequency Enhancement, and Error Correction Blocks,” *Applied Energy*, vol. 376, p. 124308, 2024.',
    'J. Shi, W. Zhang, Y. Bao, D. W. Gao, and Z. Wang, “Load Forecasting of Electric Vehicle Charging Stations: Attention Based Spatiotemporal Multi-Graph Convolutional Networks,” *IEEE Transactions on Smart Grid*, vol. 15, no. 3, pp. 3016–3027, 2024.',
    'H. Li, D. Liang, Y. Zhou, Y. Shi, D. Feng, and S. Shi, “Spatiotemporal Charging Demand Models for Electric Vehicles Considering User Strategies,” *Frontiers in Energy Research*, vol. 10, p. 1013154, 2022.',
    'Y. Zhao, Z. Wang, Z. J. M. Shen, and F. Sun, “Data-Driven Framework for Large-Scale Prediction of Charging Energy in Electric Vehicles,” *Applied Energy*, vol. 282, p. 116175, 2021.',
    'J. Zhang et al., “Charging Demand Prediction in Beijing Based on Real-World Electric Vehicle Data,” *Journal of Energy Storage*, vol. 57, p. 106294, 2023.',
    'Y. S. Liu, M. Tayarani, and H. O. Gao, “An Activity-Based Travel and Charging Behavior Model for Simulating Battery Electric Vehicle Charging Demand,” *Energy*, vol. 258, p. 124938, 2022.',
    'L. Zhang et al., “OD Matrix Based Spatiotemporal Distribution of EV Charging Load Prediction,” *Power System Protection and Control*, vol. 49, pp. 82–91, 2021.',
    # ---- method references appended for the full paper ----
    'M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise,” in *Proc. 2nd Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 1996, pp. 226–231.',
    'A. C. Cameron and P. K. Trivedi, *Regression Analysis of Count Data*, 2nd ed. Cambridge, U.K.: Cambridge University Press, 2013.',
    'A. C. Cameron and P. K. Trivedi, “Regression-Based Tests for Overdispersion in the Poisson Model,” *Journal of Econometrics*, vol. 46, no. 3, pp. 347–364, 1990.',
    'F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.',
    'S. Seabold and J. Perktold, “Statsmodels: Econometric and Statistical Modeling with Python,” in *Proc. 9th Python in Science Conf. (SciPy)*, 2010, pp. 92–96.',
    'City of Palo Alto, “Electric Vehicle Charging Station Usage, July 2011 – December 2020,” City of Palo Alto Open Data. Mirrored on Kaggle as “EV Charging Station Usage of California City” (kaggle.com/datasets/venkatsairo4899/ev-charging-station-usage-of-california-city) and on GitHub (github.com/andriysavcyn/palo-alto-ev-charging), accessed Oct. 2026.',
    'P. A. W. Lewis and G. S. Shedler, “Simulation of Nonhomogeneous Poisson Processes by Thinning,” *Naval Research Logistics Quarterly*, vol. 26, no. 3, pp. 403–413, 1979.',
    'D. Gross, J. F. Shortle, J. M. Thompson, and C. M. Harris, *Fundamentals of Queueing Theory*, 4th ed. Hoboken, NJ, USA: Wiley, 2008.',
    'C. Czado, T. Gneiting, and L. Held, “Predictive Model Assessment for Count Data,” *Biometrics*, vol. 65, no. 4, pp. 1254–1261, 2009.',
    'J. Sander, M. Ester, H.-P. Kriegel, and X. Xu, “Density-Based Clustering in Spatial Databases: The Algorithm GDBSCAN and Its Applications,” *Data Mining and Knowledge Discovery*, vol. 2, no. 2, pp. 169–194, 1998.',
    'G. Ke et al., “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017, pp. 3146–3154.',
]
