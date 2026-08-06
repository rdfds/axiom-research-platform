"""
Explore M&A and Deal Data in WRDS
=================================
Check what deal/transaction data is available for V1 analog retrieval.

Run this to understand what M&A data you have access to.
"""

import sys
import wrds

def main():
    if len(sys.argv) > 1:
        WRDS_USERNAME = sys.argv[1]
    else:
        WRDS_USERNAME = input("Enter your WRDS username: ")

    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    print("Connected!\n")

    # Check potential M&A data sources
    libraries_to_check = [
        ('tfn', 'Thomson Financial - insider/institutional'),
        ('dealscan', 'DealScan - syndicated loans'),
        ('ciq', 'Capital IQ'),
        ('ciqsamp', 'Capital IQ Sample'),
        ('ciqsamp_transactions', 'Capital IQ Transactions Sample'),
        ('ciqsamp_keydev', 'Capital IQ Key Developments'),
        ('sdc', 'SDC Platinum'),
        ('sdcm', 'SDC M&A'),
        ('tr', 'Thomson Reuters'),
        ('tr_dealscan', 'TR DealScan'),
        ('tr_insiders', 'TR Insiders'),
    ]

    all_libraries = db.list_libraries()
    if hasattr(all_libraries, 'values'):
        all_libraries = list(all_libraries.values.flatten()) if hasattr(all_libraries.values, 'flatten') else list(all_libraries.values)

    print("="*70)
    print("CHECKING DEAL/TRANSACTION DATA SOURCES")
    print("="*70)

    for lib, desc in libraries_to_check:
        if lib in all_libraries:
            print(f"\n✓ {lib} ({desc})")
            tables = db.list_tables(library=lib)
            if hasattr(tables, 'values'):
                tables = list(tables.values.flatten())
            print(f"  Tables ({len(tables)}): {tables[:15]}")
            if len(tables) > 15:
                print(f"  ... and {len(tables) - 15} more")
        else:
            print(f"\n✗ {lib} - not available")

    # Deep dive into DealScan (loan data)
    print("\n" + "="*70)
    print("DEALSCAN DEEP DIVE (Loan/Credit Facility Data)")
    print("="*70)

    if 'dealscan' in all_libraries:
        # Check key tables
        key_tables = ['facility', 'package', 'company', 'borrowerbase']
        for table in key_tables:
            try:
                cols = db.describe_table(library='dealscan', table=table)
                print(f"\ndealscan.{table}:")
                if hasattr(cols, 'head'):
                    print(cols[['name', 'type']].head(10).to_string(index=False))
                else:
                    print(f"  Columns: {cols[:10]}")
            except Exception as e:
                print(f"  Could not describe {table}: {e}")

        # Sample query
        print("\nSample from dealscan.facility:")
        try:
            sample = db.raw_sql("""
                SELECT *
                FROM dealscan.facility
                LIMIT 5
            """)
            print(sample.head())
        except Exception as e:
            print(f"  Query failed: {e}")

    # Check Capital IQ transactions if available
    print("\n" + "="*70)
    print("CAPITAL IQ DATA CHECK")
    print("="*70)

    ciq_libs = [lib for lib in all_libraries if 'ciq' in lib.lower()]
    print(f"CIQ-related libraries: {ciq_libs}")

    for lib in ciq_libs:
        if 'transaction' in lib.lower() or 'keydev' in lib.lower():
            print(f"\n--- {lib} ---")
            tables = db.list_tables(library=lib)
            if hasattr(tables, 'values'):
                tables = list(tables.values.flatten())
            print(f"Tables: {tables}")

            # Try to get sample
            if tables:
                try:
                    sample = db.raw_sql(f"SELECT * FROM {lib}.{tables[0]} LIMIT 3")
                    print(f"\nSample from {lib}.{tables[0]}:")
                    print(sample)
                except Exception as e:
                    print(f"  Could not query: {e}")

    # Check for actual M&A transaction tables
    print("\n" + "="*70)
    print("SEARCHING FOR M&A TRANSACTION TABLES")
    print("="*70)

    # Look for tables with 'merge', 'acqui', 'deal', 'transaction' in name
    ma_keywords = ['merge', 'acqui', 'deal', 'transaction', 'takeover', 'mna', 'm_a']

    found_tables = []
    for lib in ['comp', 'tfn', 'ciq', 'ciqsamp']:
        if lib in all_libraries:
            tables = db.list_tables(library=lib)
            if hasattr(tables, 'values'):
                tables = list(tables.values.flatten())

            for table in tables:
                if any(kw in table.lower() for kw in ma_keywords):
                    found_tables.append(f"{lib}.{table}")

    if found_tables:
        print(f"Found potential M&A tables: {found_tables}")
        for tbl in found_tables[:5]:
            lib, table = tbl.split('.')
            try:
                cols = db.describe_table(library=lib, table=table)
                print(f"\n{tbl}:")
                if hasattr(cols, 'head'):
                    print(cols[['name', 'type']].head(8).to_string(index=False))
            except Exception as e:
                print(f"  Could not describe: {e}")
    else:
        print("No obvious M&A tables found by name search.")

    # Check Compustat for acquisition-related data
    print("\n" + "="*70)
    print("COMPUSTAT M&A-RELATED FIELDS")
    print("="*70)

    print("Checking comp.funda for acquisition-related fields...")
    try:
        # AQC = Acquisitions, AQEPS = Acquisition diluted EPS
        sample = db.raw_sql("""
            SELECT gvkey, datadate, conm, aqc, sale
            FROM comp.funda
            WHERE aqc IS NOT NULL AND aqc > 0
            ORDER BY aqc DESC
            LIMIT 10
        """)
        print("Companies with acquisitions (aqc field):")
        print(sample)
    except Exception as e:
        print(f"  Query failed: {e}")

    db.close()
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
For V1 M&A analog retrieval, your options are:

1. DealScan - Good for LOAN transactions (refinancing, credit facilities)
   - Useful for "refinancing pressure" signal
   - NOT for M&A deals

2. Compustat AQC field - Shows acquisition spending
   - Can identify acquirers and rough timing
   - But no deal details (target, price, terms)

3. Capital IQ (if ciq or ciqsamp_transactions accessible)
   - May have detailed M&A data
   - Check tables above

4. Manual curation (fallback)
   - Use LSEG Workspace or SDC Platinum at JHU
   - Build 50-100 deal cases for industrials
    """)


if __name__ == "__main__":
    main()
