# Task C - note on the board figure

**Reader.** The National Grid Authority board, splitting the reliability budget across 24 districts.

**Message.** Per household, supply is least reliable in Lashkar (D12), Kotla (D11) and Mehrabad (D13), about 50 interruptions per 1,000 households a year against 43.7 nationally, and the biggest districts are not the worst served.

**Parameters, each with a reason**

- **Denominator: billed households** (interruptions per 1,000 billed households, 12 months). Interruptions are counted on the feeders that serve billed connections, so this is closest to "how often does a connected home lose supply". Population would also count the 10-33% of people who aren't connected. The D21 household count was repaired first (182 -> 182,000); with the raw value D21 would have shown 47,907 per 1,000.
- **Classes: natural breaks (Jenks), 5 classes**, edges 30.4 / 33.5 / 36.2 / 42.7 / 45.8 / 50.9, printed in the legend. GVF 0.959 (equal interval ties, quantile 0.899). Quantile would split the 40-42 cluster into three colours that mean nothing.
- **Scheme: sequential Blues**, one hue light to dark, because the rate has one direction (more is worse). No rainbow. Checked with `vizlib.cvd_report`.
- **Projection: Lambert cylindrical equal-area**, one reference latitude for the whole map, because on a choropleth area is the channel.
- **Bar panel: sorted by rate, zero baseline**, coloured with the map's classes, with the count printed beside each rate. It is there because the map gives the five most populous districts (59.9% of the people) only 12.6% of the ink.
- **Window: the 12 months in the register.** It is the only period the file has, so I didn't choose it, but it is stated on the axis.
- **Suppression: none.** Rule set before looking: no rate from fewer than 20 interruptions or 10,000 households. The smallest counts are 3,234 interruptions (D16) and 76,721 households (D14). With counts this large, the Poisson funnel puts 8 districts above the national 95% band, 14 below and 2 inside, so the ranking is not noise.

**The counterfactual.** A raw-count map is a map of households (r = +0.991). It would have pointed at **Qilabad (D17), Mehrabad (D13) and Zhobara (D24)**. Qilabad is 19th of 24 per household and Zhobara 12th; both are below the national rate. A budget split on that map would take money **away from Lashkar (D12) and Kotla (D11)**, the two worst-served districts (8th and 5th by count), and from Rohtas West, Chakwa and Ushkan (4th-6th per household, 13th-17th by count). Mehrabad is the only district on both lists, and not because it is big: per household, rate and size are almost unrelated (r = +0.175). It is there because it is poor: joint-lowest income in the federation ($2,600 a head, with Virkabad), and across districts the rate falls as income rises (r = -0.71).

**One caveat for the board.** This is how *often* supply fails. Mehrabad's interruptions are short (102 minutes on average, against 205-208 in Lashkar and Kotla), so on interruption *minutes* per household it drops to 17th of 22 (D09 and D20 have no duration). Lashkar and Kotla are 1st and 2nd on both measures, so they are the safe call.
