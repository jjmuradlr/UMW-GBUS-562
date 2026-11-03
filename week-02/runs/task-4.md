Model: claude-sonnet-5

### System
You are a retail operations analyst helping Wildframe Coffee Roasters, a specialty coffee roaster, make careful, numbers-based decisions.

### You
CONTEXT: Wildframe can run one four-week pop-up stand this spring. The total rent budget is $5,000 for all four weeks. Wildframe mainly sells whole-bean bags and a monthly subscription, and its customers are environmentally conscious urban professionals. Here is weekly data from comparable past events:

Riverside Farmers Market: 9,000 visitors per week; 2% of visitors buy; average sale $14 (mostly whole-bean bags); rent $900 per week.
Union Station concourse: 30,000 visitors per week; 0.5% of visitors buy; average sale $6 (mostly single cups); rent $2,000 per week.
Campus library atrium: 6,000 visitors per week; 3% of visitors buy; average sale $5 (single cups); rent $600 per week.

TASK: Decide which location the pop-up should go to. Think through this step by step and show your reasoning before giving the answer.

For each location, show:
1. Weekly buyers (visitors × purchase rate)
2. Weekly revenue (buyers × average sale)
3. Four-week revenue
4. Four-week rent, and whether it fits the $5,000 budget
5. Four-week revenue minus rent

Then compare the locations, including which one best fits Wildframe's business (whole-bean bags and subscriptions versus single cups), and choose one.

FORMAT: Show the calculations for each location, then a short comparison. The very last line of the response must be a single line that starts with "Recommendation:" followed only by the location's name. Write nothing after that line.

CONSTRAINTS:
- Use only the data given. Do not invent costs such as staffing or coffee costs. If you mention them, label them as outside the data.
- Any location whose four-week rent exceeds $5,000 is ruled out.
- Double-check every calculation.

### Claude
# Wildframe Coffee Pop-Up: Location Analysis

## 1. Riverside Farmers Market

- **Weekly buyers:** 9,000 × 2% = 180 buyers
- **Weekly revenue:** 180 × $14 = $2,520
- **Four-week revenue:** $2,520 × 4 = $10,080
- **Four-week rent:** $900 × 4 = $3,600 → **Fits budget** ($3,600 ≤ $5,000)
- **Four-week revenue minus rent:** $10,080 − $3,600 = **$6,480**

## 2. Union Station Concourse

- **Weekly buyers:** 30,000 × 0.5% = 150 buyers
- **Weekly revenue:** 150 × $6 = $900
- **Four-week revenue:** $900 × 4 = $3,600
- **Four-week rent:** $2,000 × 4 = $8,000 → **Exceeds budget** ($8,000 > $5,000) — **RULED OUT**
- **Four-week revenue minus rent:** Not applicable (disqualified)

## 3. Campus Library Atrium

- **Weekly buyers:** 6,000 × 3% = 180 buyers
- **Weekly revenue:** 180 × $5 = $900
- **Four-week revenue:** $900 × 4 = $3,600
- **Four-week rent:** $600 × 4 = $2,400 → **Fits budget** ($2,400 ≤ $5,000)
- **Four-week revenue minus rent:** $3,600 − $2,400 = **$1,200**

## Comparison

| Location | 4-Wk Revenue | 4-Wk Rent | Net | Budget OK? |
|---|---|---|---|---|
| Riverside | $10,080 | $3,600 | **$6,480** | Yes |
| Union Station | $3,600 | $8,000 | n/a | **No — ruled out** |
| Campus Library | $3,600 | $2,400 | $1,200 | Yes |

Union Station is disqualified outright since its rent alone ($8,000) exceeds the entire $5,000 budget.

Between the two remaining options, Riverside Farmers Market generates more than 5x the net return of Campus Library ($6,480 vs. $1,200), despite similar buyer counts per week. This is driven by Riverside's much higher average sale ($14 vs. $5), because Riverside's traffic skews toward **whole-bean bag purchases** — Wildframe's core product and the natural on-ramp to its subscription offering. Campus Library's traffic is mostly **single-cup buyers**, which generates foot traffic but doesn't align as well with promoting bags or subscriptions, and produces far less revenue overall.

Riverside also leaves $1,400 of the rent budget unused, which could (outside this data) be allocated to signage, subscription sign-up incentives, or sampling — all of which fit its environmentally-conscious, urban-professional customer base better than a single-cup-dominated transit venue.

**Note:** Staffing, coffee/COGS, and promotional costs are not included in this analysis, as they were not provided in the data.

Recommendation: Riverside Farmers Market
