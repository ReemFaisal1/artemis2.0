# aggregate to one row per (mc_id, time_slice)
agg = (df.groupby(['mc_id','time_slice'])
         .agg({
             'Type': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
             'range_m': 'median',
             'length_m': 'median',
             'RCSinst_dB': 'mean',
             'SNRinst_dB': 'mean',
         })
         .reset_index())

# now build slice 0 and 1 tables with equal length by pivoting on mc_id
slice0 = agg[agg['time_slice'] == 0].set_index('mc_id')
slice1 = agg[agg['time_slice'] == 1].set_index('mc_id')

# keep only mc_ids that appear in BOTH slices (important)
common_ids = slice0.index.intersection(slice1.index)
slice0 = slice0.loc[common_ids]
slice1 = slice1.loc[common_ids]

# build dbn_df with MultiIndex columns
dbn_df = pd.concat([
    slice0[['Type','range_m','length_m','RCSinst_dB','SNRinst_dB']].rename(columns=lambda c: (c,0)),
    slice1[['Type','range_m','length_m','RCSinst_dB','SNRinst_dB']].rename(columns=lambda c: (c,1)),
], axis=1)

dbn_df.columns = pd.MultiIndex.from_tuples(dbn_df.columns)
dbn_df = dbn_df.reset_index(drop=True)