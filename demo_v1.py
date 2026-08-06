#!/usr/bin/env python
"""
Axiom V1 Demo
=============
Demonstrates the complete V1 system:
1. As-of snapshot building (no lookahead bias)
2. State profile computation (7 signals)
3. Market regime classification
4. Analog retrieval from historical M&A transactions

Run: python demo_v1.py
"""

import sys
sys.path.insert(0, '.')

from src.snapshot import AsOfSnapshotBuilder
from src.signals import SignalEngine
from src.regimes import RegimeClassifier
from src.analogs import AnalogRetriever


def print_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main():
    print_header("AXIOM V1 DEMO")
    print("Decision Intelligence for Capital Allocation")

    # Initialize components
    print("\n[1/4] Initializing components...")
    snapshot = AsOfSnapshotBuilder()
    engine = SignalEngine(snapshot)
    regime = RegimeClassifier()

    # Query date and universe
    as_of_date = '2023-06-30'

    print_header(f"STEP 1: GET UNIVERSE AS OF {as_of_date}")
    print("Using filing dates (rdq) to avoid lookahead bias...")

    universe = snapshot.get_universe_snapshot(
        as_of_date,
        min_assets=500,   # $500M+ assets
        min_revenue=100   # $100M+ quarterly revenue
    )

    print(f"\nFiltered universe: {len(universe)} industrial companies")
    print("\nTop 10 by assets:")
    top10 = universe.nlargest(10, 'atq')[['gvkey', 'conm', 'tic', 'atq', 'revtq']]
    print(top10.to_string(index=False))

    # Select a sample company for deep dive
    sample = universe.iloc[5]  # Pick a mid-sized company
    gvkey = sample['gvkey']
    company_name = sample['conm']
    ticker = sample['tic']

    print_header(f"STEP 2: COMPUTE STATE PROFILE FOR {company_name}")

    profile = engine.compute_state_profile(gvkey, as_of_date)

    if profile:
        print(f"\n{company_name} ({ticker})")
        print(f"GVKEY: {gvkey}")
        print(f"As of: {as_of_date}")
        print(f"\n{'Signal':<30} {'Score':>8} {'Interpretation':<25}")
        print("-" * 65)

        interpretations = {
            'balance_sheet_optionality': lambda s: 'Strong' if s > 70 else 'Moderate' if s > 40 else 'Weak',
            'growth_momentum': lambda s: 'High Growth' if s > 70 else 'Stable' if s > 40 else 'Declining',
            'margin_trend': lambda s: 'Expanding' if s > 60 else 'Stable' if s > 40 else 'Compressing',
            'valuation_dislocation': lambda s: 'Undervalued' if s > 60 else 'Fair Value' if s > 40 else 'Premium',
            'refinancing_pressure': lambda s: 'Low Risk' if s > 70 else 'Moderate' if s > 40 else 'High Risk',
            'size_factor': lambda s: 'Large' if s > 60 else 'Mid-Cap' if s > 30 else 'Small',
            'asset_intensity': lambda s: 'Asset Light' if s > 60 else 'Moderate' if s > 40 else 'Asset Heavy',
        }

        for sig_name, sig_data in profile['signals'].items():
            score = sig_data['score']
            interp = interpretations.get(sig_name, lambda s: '')(score)
            print(f"{sig_name.replace('_', ' ').title():<30} {score:>6.1f}/100  {interp:<25}")

        print("-" * 65)
        print(f"{'COMPOSITE SCORE':<30} {profile['composite_score']:>6.1f}/100")

        # Show key components
        print("\n📊 Key Metrics:")
        bs = profile['signals']['balance_sheet_optionality']['components']
        print(f"   Net Debt/EBITDA: {bs.get('leverage_ratio', 'N/A')}")
        print(f"   Cash/Assets: {bs.get('cash_to_assets', 0):.1%}")

        growth = profile['signals']['growth_momentum']['components']
        print(f"   YoY Revenue Growth: {growth.get('yoy_revenue_growth', 0):.1%}" if growth.get('yoy_revenue_growth') else "   YoY Revenue Growth: N/A")

        val = profile['signals']['valuation_dislocation']['components']
        print(f"   EV/EBITDA: {val.get('ev_ebitda', 'N/A')}x")

    print_header(f"STEP 3: CLASSIFY MARKET REGIME")

    regime_result = regime.classify_regime(as_of_date)
    chars = regime.get_regime_characteristics(regime_result['regime'])

    print(f"\nRegime: {regime_result['regime']}")
    print(f"Description: {regime_result['description']}")
    print(f"\nRegime Characteristics:")
    print(f"   Typical EV/EBITDA: {chars['typical_ev_ebitda_range'][0]}x - {chars['typical_ev_ebitda_range'][1]}x")
    print(f"   Expected Premium: {chars['expected_premium_range'][0]}% - {chars['expected_premium_range'][1]}%")
    print(f"   Deal Activity: {chars['deal_activity']}")
    print(f"   Financing: {chars['financing_availability']}")

    # Regime-adjusted expectations
    adjusted = regime.get_regime_adjusted_expectations(profile, regime_result['regime'])
    print(f"\n📈 Regime-Adjusted Valuation Range:")
    print(f"   Low: {adjusted['expected_ev_ebitda']['low']:.1f}x EBITDA")
    print(f"   Base: {adjusted['expected_ev_ebitda']['base']:.1f}x EBITDA")
    print(f"   High: {adjusted['expected_ev_ebitda']['high']:.1f}x EBITDA")

    print_header("STEP 4: FIND HISTORICAL ANALOGS")

    print("Searching for M&A transactions with similar state profiles...")
    print("(This uses the Capital IQ transaction sample data)")

    try:
        retriever = AnalogRetriever(signal_engine=engine)

        analogs = retriever.find_analogs(
            profile,
            n_analogs=5,
            min_similarity=0.5
        )

        if analogs:
            print(f"\nFound {len(analogs)} analog transactions:\n")
            for i, analog in enumerate(analogs, 1):
                print(f"{i}. {analog['target_name']}")
                print(f"   Acquirer: {analog['acquirer_name']}")
                print(f"   Date: {analog['deal_date'].strftime('%Y-%m-%d') if analog['deal_date'] else 'N/A'}")
                print(f"   Value: ${analog['deal_value']:,.0f}M" if analog['deal_value'] else "   Value: N/A")
                print(f"   Similarity: {analog['similarity_score']:.1%}")
                print()
        else:
            print("\nNo closely matching analogs found in the sample data.")
            print("(The CIQ sample has limited coverage - production would use full dataset)")
    except Exception as e:
        print(f"\nNote: Analog search encountered an issue: {e}")
        print("(This is expected if company IDs don't align between datasets)")

    print_header("V1 DEMO COMPLETE")
    print("""
What we demonstrated:
✓ As-of snapshot builder (no lookahead bias via rdq dates)
✓ 7 interpretable signals forming a "state profile"
✓ Market regime classification (LOOSE/SELECTIVE/TIGHT)
✓ Regime-adjusted valuation expectations
✓ Historical analog retrieval framework

Next steps for V1:
- Expand analog matching to use more transaction data
- Add TSR outcome calculations from price data
- Build the narrative/defense layer
- Create a simple web interface
    """)


if __name__ == "__main__":
    main()
