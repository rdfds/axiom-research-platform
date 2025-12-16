"""
WRDS Connection Test & Data Exploration
=======================================
Run this script to:
1. Test your WRDS connection
2. See what databases you have access to
3. Explore Compustat and CRSP tables for V1

First time setup:
- pip install wrds
- You'll be prompted for your WRDS username/password on first run
- After that, credentials are stored in ~/.pgpass

Usage:
  python 01_test_wrds_connection.py YOUR_WRDS_USERNAME
"""

import sys
import wrds

def main():
    # Get username from command line or prompt
    if len(sys.argv) > 1:
        WRDS_USERNAME = sys.argv[1]
    else:
        WRDS_USERNAME = input("Enter your WRDS username: ")

    print("\n" + "="*60)
    print("CONNECTING TO WRDS...")
    print("="*60)

    try:
        db = wrds.Connection(wrds_username=WRDS_USERNAME)
        print("✓ Connected successfully!\n")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nMake sure you have:")
        print("  1. A valid WRDS account (not a class/daypass account)")
        print("  2. Installed the wrds package: pip install wrds")
        return

    # List all available libraries (databases)
    print("="*60)
    print("AVAILABLE LIBRARIES (DATABASES)")
    print("="*60)
    libraries = db.list_libraries()

    # Handle both list and DataFrame return types
    if hasattr(libraries, 'values'):
        libraries = libraries.values.tolist() if hasattr(libraries.values, 'tolist') else list(libraries.values)
    libraries = list(libraries) if not isinstance(libraries, list) else libraries

    print(f"You have access to {len(libraries)} libraries:\n")

    # Group by category for readability
    key_libraries = ['comp', 'crsp', 'tfn', 'tr', 'sdcm', 'dealscan']

    print("Key libraries for Axiom V1:")
    for lib in key_libraries:
        if lib in libraries:
            print(f"  ✓ {lib}")
        else:
            print(f"  ✗ {lib} (not available)")

    print(f"\nAll libraries (first 50): {sorted(libraries)[:50]}")
    if len(libraries) > 50:
        print(f"  ... and {len(libraries) - 50} more")

    # Check Compustat access (fundamentals)
    print("\n" + "="*60)
    print("COMPUSTAT TABLES (Fundamentals)")
    print("="*60)

    if 'comp' in libraries:
        comp_tables = db.list_tables(library='comp')
        if hasattr(comp_tables, 'values'):
            comp_tables = list(comp_tables.values.flatten()) if hasattr(comp_tables.values, 'flatten') else list(comp_tables.values)
        comp_tables = list(comp_tables) if not isinstance(comp_tables, list) else comp_tables

        print(f"Found {len(comp_tables)} tables in comp library")

        # Key tables for V1
        key_tables = ['fundq', 'funda', 'company', 'security']
        print("\nKey tables for V1:")
        for table in key_tables:
            if table in comp_tables:
                print(f"  ✓ comp.{table}")
            else:
                print(f"  ✗ comp.{table}")

        # Describe fundq (quarterly fundamentals)
        if 'fundq' in comp_tables:
            print("\n--- comp.fundq columns (quarterly fundamentals) ---")
            fundq_cols = db.describe_table(library='comp', table='fundq')

            # Key columns for V1
            v1_columns = [
                ('gvkey', 'Company identifier'),
                ('datadate', 'Fiscal period end date'),
                ('rdq', 'Report date (FILING DATE - critical for as-of!)'),
                ('fyearq', 'Fiscal year'),
                ('fqtr', 'Fiscal quarter'),
                ('revtq', 'Revenue'),
                ('oibdpq', 'Operating income before depreciation (EBITDA proxy)'),
                ('atq', 'Total assets'),
                ('ltq', 'Total liabilities'),
                ('ceqq', 'Common equity'),
                ('dlttq', 'Long-term debt'),
                ('dlcq', 'Debt in current liabilities'),
                ('cheq', 'Cash and short-term investments'),
                ('xintq', 'Interest expense'),
                ('capxy', 'Capital expenditures'),
            ]

            print("\nKey columns for V1 signals:")
            if hasattr(fundq_cols, 'values'):
                available_cols = list(fundq_cols['name'].values)
            else:
                available_cols = [row[0] for row in fundq_cols] if fundq_cols else []

            for col, desc in v1_columns:
                status = "✓" if col in available_cols else "✗"
                print(f"  {status} {col}: {desc}")
    else:
        print("✗ Compustat not available in your subscription")

    # Check CRSP access (stock prices)
    print("\n" + "="*60)
    print("CRSP TABLES (Stock Prices)")
    print("="*60)

    if 'crsp' in libraries:
        crsp_tables = db.list_tables(library='crsp')
        if hasattr(crsp_tables, 'values'):
            crsp_tables = list(crsp_tables.values.flatten()) if hasattr(crsp_tables.values, 'flatten') else list(crsp_tables.values)
        crsp_tables = list(crsp_tables) if not isinstance(crsp_tables, list) else crsp_tables

        print(f"Found {len(crsp_tables)} tables in crsp library")

        key_tables = ['dsf', 'msf', 'dsi', 'msi']
        print("\nKey tables for V1:")
        for table in key_tables:
            if table in crsp_tables:
                print(f"  ✓ crsp.{table}")
            else:
                print(f"  ✗ crsp.{table}")
    else:
        print("✗ CRSP not available in your subscription")

    # Check for M&A / Deal data
    print("\n" + "="*60)
    print("M&A / DEAL DATA")
    print("="*60)

    ma_libraries = ['tfn', 'tr', 'sdcm', 'sdc', 'dealscan']
    found_ma = False
    for lib in ma_libraries:
        if lib in libraries:
            print(f"✓ Found {lib} - checking tables...")
            tables = db.list_tables(library=lib)
            if hasattr(tables, 'values'):
                tables = list(tables.values.flatten())[:10]
            else:
                tables = list(tables)[:10]
            print(f"  Tables: {tables}...")
            found_ma = True

    if not found_ma:
        print("✗ No M&A databases found (tfn, tr, sdcm, dealscan)")
        print("  You'll need to use SDC Platinum on JHU terminals or manual curation")

    # Sample query to verify data access
    print("\n" + "="*60)
    print("SAMPLE DATA PULL (Testing Access)")
    print("="*60)

    if 'comp' in libraries:
        print("Pulling 5 sample rows from comp.fundq...")
        try:
            sample = db.raw_sql("""
                SELECT gvkey, datadate, rdq, revtq, atq
                FROM comp.fundq
                WHERE datadate >= '2020-01-01'
                LIMIT 5
            """)
            print(sample)
            print("\n✓ Data access confirmed!")
        except Exception as e:
            print(f"✗ Query failed: {e}")

    # Close connection
    db.close()
    print("\n" + "="*60)
    print("CONNECTION CLOSED")
    print("="*60)

    print("\nNext steps:")
    print("  1. Run 02_pull_fundamentals.py to download historical data")
    print("  2. Check if you have M&A data access above")
    print("  3. If no M&A data, plan to use SDC Platinum at JHU library")


if __name__ == "__main__":
    main()
