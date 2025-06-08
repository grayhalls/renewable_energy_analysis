import streamlit as st
import kagglehub
import os
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Sustainable Energy Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state='collapsed'
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1E88E5;
        font-size: 2.5rem !important;
        margin-bottom: 2rem !important;
    }
    .feature-description {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Global Sustainable Energy Analysis")

with st.expander("📊 Key Features and Metrics", expanded=True):
    col_feat1, col_feat2 = st.columns(2)

    with col_feat1:
        st.markdown("### Basic Information")
        st.markdown("- **Entity**: Country or region name")
        st.markdown("- **Year**: Data year (2000-2019)")
        st.markdown("- **GDP per capita**: Gross domestic product per person")
        st.markdown("- **GDP growth**: Annual GDP growth rate (%)")
        st.markdown("- **Density**: Population density (P/Km²)")
        st.markdown("- **Land Area**: Total land area (Km²)")
        st.markdown("- **Location**: Latitude and Longitude coordinates")

        st.markdown("### Energy Access")
        st.markdown("- **Access to electricity**: % of population with electricity access")
        st.markdown("- **Access to clean fuels**: % of population with clean cooking fuels")
        st.markdown("- **Renewable capacity per capita**: Installed renewable energy capacity per person")

    with col_feat2:
        st.markdown("### Energy Generation")
        st.markdown("- **Electricity from fossil fuels**: Generation in TWh")
        st.markdown("- **Electricity from nuclear**: Generation in TWh")
        st.markdown("- **Electricity from renewables**: Generation in TWh")
        st.markdown("- **Low-carbon electricity**: % of electricity from nuclear and renewables")

        st.markdown("### Energy Consumption & Efficiency")
        st.markdown("- **Primary energy consumption**: kWh per capita")
        st.markdown("- **Energy intensity**: MJ per $2017 PPP GDP")
        st.markdown("- **Renewable energy share**: % of total final energy consumption")
        st.markdown("- **Renewables**: % of equivalent primary energy")

        st.markdown("### Environmental Impact")
        st.markdown("- **CO₂ emissions**: Metric tons per capita")
        st.markdown("- **Financial flows**: US$ to developing countries for clean energy")

# Load data
@st.cache_data
def load_data():
    energy_path = kagglehub.dataset_download("anshtanwar/global-data-on-sustainable-energy")
    energy_file = [f for f in os.listdir(energy_path) if f.endswith('.csv')][0]

    continent_path = kagglehub.dataset_download("andradaolteanu/country-mapping-iso-continent-region")
    continent_file = [f for f in os.listdir(continent_path) if f.endswith('.csv')][0]

    energy_df = pd.read_csv(os.path.join(energy_path, energy_file))
    continent_df = pd.read_csv(os.path.join(continent_path, continent_file))
    continent_df = continent_df[['name', 'region', 'sub-region']]

    energy_df = energy_df.merge(continent_df, left_on='Entity', right_on='name', how='left')

    # st.write(energy_df.columns)
    energy_df.rename(columns={
        'Access to electricity (% of population)': 'Access to electricity (%)',
        'Access to clean fuels for cooking': 'Access to clean fuels (%)',
        'Renewable energy share in the total final energy consumption (%)': 'Renewable energy share (%)', 
        'Electricity from fossil fuels (TWh)': 'Electricity from fossil fuels (TWh)',
        'Electricity from nuclear (TWh)': 'Electricity from nuclear (TWh)',
        'Electricity from renewables (TWh)': 'Electricity from renewables (TWh)',
        'Low-carbon electricity (% electricity)': 'Low-carbon electricity (%)',
        'Primary energy consumption per capita (kWh/person)': 'Primary energy consumption (kWh/person)',
        'Energy intensity level of primary energy (MJ/$2017 PPP GDP)': 'Energy intensity (MJ/$2017 PPP GDP)',
        'Value_co2_emissions_kt_by_country': 'CO2 emissions (metric tons per capita)',
        'Renewables (% equivalent primary energy)': 'Renewables (% primary energy)',
        'Financial flows to developing countries (US $)': 'Financial flows (US$)',
        'gdp_per_capita': 'GDP per capita', 
        'gpd_growth': 'GDP growth (%)',
        'Density\\n(P/Km2)': 'Density (P/Km2)',
        'region': 'Continent'
    }, inplace=True)
    # st.write(energy_df.columns)
    numeric_cols = [
        'Access to electricity (%)', 'Access to clean fuels (%)', 'Renewable energy share (%)',
        'Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)',
        'Low-carbon electricity (%)', 'Primary energy consumption (kWh/person)', 'Energy intensity (MJ/$2017 PPP GDP)',
        'CO2 emissions (metric tons per capita)', 'Renewables (% primary energy)', 'Financial flows (US$)',
        'GDP per capita', 'Density (P/Km2)','Latitude', 'Longitude'
    ]
    for col in numeric_cols:
        energy_df[col] = pd.to_numeric(energy_df[col], errors='coerce')

    return energy_df

st.header("Our Project Goals 🎯")
st.markdown("""
Our primary goals for this sustainable energy analysis are to:
* **Understand Global Energy Access:** Assess the current state and trends of electricity and clean cooking fuel access worldwide.
* **Track Renewable Transition:** Analyze the pace at which continents are shifting towards renewable and low-carbon electricity, and how this impacts overall energy consumption.
* **Evaluate Environmental Impact:** Investigate the relationship between economic development, energy consumption, and CO2 emissions.
* **Identify Key Drivers:** Uncover correlations between socioeconomic factors (like GDP, population density) and progress in sustainable energy adoption.
""")

st.header("Tasks Performed ✨")
st.markdown("""
To achieve our goals, we performed the following key tasks:
* Understand global trends and energy access by creating simple line trend graphs.
* Analyze how the world is shifting in renewable energy generation by developing map visualizations.
* Investicate correlations between development, energy use, and emissions by creating a series of scatterplot correlation graphs.
""")

try:
    df = load_data()

    ## --- CLEANING DATASET & INITIAL FILTERING ---
    df.dropna(subset=['Continent'], inplace=True)
    df = df[df['Year'] != 2020] # Filter out year 2020 as it has minimal data

    # Show a subset of the data to users (for debugging/info)
    st.subheader("Subset of Dataset")
    st.write(df[(df['Year'] == 2019) & (df['Entity'].isin(['United States', 'United Kingdom', 'Japan', 'China', 'Saudi Arabia']))])

    # Sidebar filters
    st.sidebar.header("Data Filters")

    regions = sorted(df['Continent'].dropna().unique())
    selected_regions = st.sidebar.multiselect(
        "Select Continents",
        regions
    )
    if not selected_regions:
        selected_regions = regions

    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max()))
    )

    filtered_df = df[
        (df['Year'].between(year_range[0], year_range[1])) &
        (df['Continent'].isin(selected_regions))
    ]
    
    with st.expander("Data Summary"):
        st.subheader("Data Summary (Current Filters)")
        st.dataframe(filtered_df.describe(include='all'), use_container_width=True)
    
    if not filtered_df.empty:
        comparison_agg_df = filtered_df.groupby(['Year', 'Continent']).agg(
            Original_Consumption_Share=('Renewable energy share (%)', 'mean'),
            total_fossil_twh=('Electricity from fossil fuels (TWh)', 'sum'),
            total_nuclear_twh=('Electricity from nuclear (TWh)', 'sum'),
            total_renewables_twh=('Electricity from renewables (TWh)', 'sum')
        ).reset_index()

        comparison_agg_df['Total Electricity Generation (TWh)'] = \
            comparison_agg_df['total_fossil_twh'] + \
            comparison_agg_df['total_nuclear_twh'] + \
            comparison_agg_df['total_renewables_twh']

        comparison_agg_df['Calculated_Electricity_Share'] = np.where(
            comparison_agg_df['Total Electricity Generation (TWh)'] > 0,
            (comparison_agg_df['total_renewables_twh'] / comparison_agg_df['Total Electricity Generation (TWh)']) * 100,
            0
        )
        comparison_agg_df['Share_Variance (%)'] = \
            comparison_agg_df['Original_Consumption_Share'] - comparison_agg_df['Calculated_Electricity_Share']
    else:
        st.warning("No data available for selected filters. Please adjust your selections.")
        st.stop() # Stop execution if no data


    # --- Section 1 ---
    st.markdown("---")
    st.header("1. Understand Global Energy Access 💡")
    st.markdown("Access to clean and affordable energy is fundamental for development. These charts show global progress and regional disparities in basic energy access.")

    col_access1, col_access2 = st.columns(2)

    with col_access1:
        fig_access = px.line(
            filtered_df.groupby('Year')['Access to electricity (%)'].mean().reset_index(),
            x='Year',
            y='Access to electricity (%)',
            title='Global Average Access to Electricity'
        )
        st.plotly_chart(fig_access, use_container_width=True)

    with col_access2:
        fig_clean_fuels_access = px.line(
            filtered_df.groupby('Year')['Access to clean fuels (%)'].mean().reset_index(),
            x='Year',
            y='Access to clean fuels (%)',
            title='Global Average Access to Clean Cooking Fuels'
        )
        st.plotly_chart(fig_clean_fuels_access, use_container_width=True)

    st.markdown("---")
    st.subheader("Regional Disparities in Energy Access (2019)")
    col_access_bar1, col_access_bar2 = st.columns(2)

    with col_access_bar1:
        access_continent_df = filtered_df[filtered_df['Year'] == 2019].copy()
        access_continent_df.dropna(subset=['Access to electricity (%)', 'Continent'], inplace=True)
        access_continent_agg = access_continent_df.groupby('Continent')['Access to electricity (%)'].mean().reset_index()
        access_continent_agg = access_continent_agg.sort_values('Access to electricity (%)', ascending=False)
        fig_access_continent_bar = px.bar(
            access_continent_agg,
            x='Continent',
            y='Access to electricity (%)',
            color='Continent',
            title='Average Electricity Access by Continent (2019)',
            labels={'Access to electricity (%)': 'Avg. Access to Electricity (%)'}
        )
        fig_access_continent_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_access_continent_bar, use_container_width=True)

    with col_access_bar2:
        clean_fuels_continent_df = filtered_df[filtered_df['Year'] == 2019].copy()
        clean_fuels_continent_df.dropna(subset=['Access to clean fuels (%)', 'Continent'], inplace=True)
        clean_fuels_continent_agg = clean_fuels_continent_df.groupby('Continent')['Access to clean fuels (%)'].mean().reset_index()
        clean_fuels_continent_agg = clean_fuels_continent_agg.sort_values('Access to clean fuels (%)', ascending=False)
        fig_clean_fuels_continent_bar = px.bar(
            clean_fuels_continent_agg,
            x='Continent',
            y='Access to clean fuels (%)',
            color='Continent',
            title='Average Clean Fuels Access by Continent (2019)',
            labels={'Access to clean fuels (%)': 'Avg. Clean Fuels Access (%)'}
        )
        fig_clean_fuels_continent_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_clean_fuels_continent_bar, use_container_width=True)


    # --- Section 2 ---
    st.markdown("---")
    st.header("2. Track Renewable Transition ⚡")
    st.markdown("The energy transition is largely driven by decarbonizing electricity generation. These visuals show how the world's energy mix is changing and the regional progress.")

    col_area_chart, col_pie_chart = st.columns([0.7, 0.3]) # Adjust column width ratio as needed

    with col_area_chart:
        st.subheader("Global Electricity Generation Mix Over Time")
        energy_mix_df = filtered_df.groupby('Year')[
            ['Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)']
        ].sum().reset_index()
        fig_energy_mix = px.area(
            energy_mix_df,
            x='Year',
            y=['Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)'],
            labels={'value': 'Electricity Generation (TWh)', 'variable': 'Source'}
        )
        fig_energy_mix.update_layout(
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_energy_mix, use_container_width=True)

    with col_pie_chart: # This column will now contain the bar chart
        st.subheader("2019 Generation")
        pie_df_2019 = energy_mix_df[energy_mix_df['Year'] == 2019].copy()

        if not pie_df_2019.empty:
            pie_data = pie_df_2019.melt(
                id_vars=['Year'],
                value_vars=['Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)'],
                var_name='Source',
                value_name='Generation (TWh)'
            )
            pie_data['Source'] = pie_data['Source'].str.replace(' (TWh)', '', regex=False)

            fig_bar_2019 = px.bar(
                pie_data,
                x='Source',
                y='Generation (TWh)',
                color='Source',
                labels={'Generation (TWh)': 'Generation (TWh)', 'Source': 'Energy Source'}
            )
            fig_bar_2019.update_layout(
                showlegend=False, 
                margin=dict(l=10, r=10, t=50, b=10)
            )
            fig_bar_2019.update_traces(texttemplate='%{y:.2s} TWh', textposition='outside')

            st.plotly_chart(fig_bar_2019, use_container_width=True)
        else:
            st.warning("No data for 2019 Electricity Generation Mix to create pie chart.")

    st.markdown("---")
    st.subheader("Renewable Electricity Generation Share by Continent Over Time")
    fig_renewable_calculated = px.line(
        comparison_agg_df, # Reusing the pre-calculated df
        x='Year',
        y='Calculated_Electricity_Share', 
        color='Continent',
        labels={'Calculated_Electricity_Share': 'Renewable Electricity Share (%)'}
    )
    fig_renewable_calculated.update_layout(legend_title="Continent", hovermode='x unified')
    st.plotly_chart(fig_renewable_calculated, use_container_width=True)

    st.markdown("---")
    st.subheader("Renewable Energy Across the Globe (2019): Generation Volume vs. Consumption Share") 
    col_map_left, col_map_right = st.columns(2)

    with col_map_left:
        map_df_gen_2019 = filtered_df[filtered_df['Year'] == 2019].copy()
       
        map_df_gen_2019.dropna(subset=['Entity', 'Electricity from renewables (TWh)'], inplace=True)
        
        fig_map_gen_2019 = px.choropleth(
            map_df_gen_2019,
            locations='Entity',
            locationmode='country names',
            color='Electricity from renewables (TWh)',
            hover_name='Entity',
            color_continuous_scale=px.colors.sequential.Plasma,
            title='Electricity from Renewables by Country', 
            projection='natural earth' 
        )
        fig_map_gen_2019.update_geos(showocean=True, oceancolor="LightBlue", showland=True, landcolor="LightGray", showcountries=True, countrycolor="DarkGray")
        fig_map_gen_2019.update_layout(coloraxis_colorbar=dict(title="Renewables (TWh)")) 
        st.plotly_chart(fig_map_gen_2019, use_container_width=True)


    with col_map_right:
        map_df_share_2019 = filtered_df[filtered_df['Year'] == 2019].copy()
        map_df_share_2019.dropna(subset=['Entity', 'Renewable energy share (%)'], inplace=True)
        
        fig_map_share_2019 = px.choropleth(
            map_df_share_2019,
            locations='Entity',
            locationmode='country names',
            color='Renewable energy share (%)',
            hover_name='Entity',
            color_continuous_scale=px.colors.sequential.Plasma,
            title='Renewable Energy Share by Country',
            projection='natural earth'
        )
        fig_map_share_2019.update_geos(showocean=True, oceancolor="LightBlue", showland=True, landcolor="LightGray", showcountries=True, countrycolor="DarkGray")
        fig_map_share_2019.update_layout(coloraxis_colorbar=dict(title="Renewable Energy Share (%)"))
        st.plotly_chart(fig_map_share_2019, use_container_width=True)

    # --- Section 3---
    st.markdown("---")
    st.header("3. Evaluate Environmental Impact 📈💨")
    st.markdown("As economies grow and energy consumption rises, managing CO2 emissions is critical. These graphs examine the relationship between development, energy use, and carbon footprint.")

    st.subheader("Low-Carbon Electricity vs. CO2 Emissions by Country")
    entity_summary_df = filtered_df.groupby('Entity')[
        ['CO2 emissions (metric tons per capita)', 'Low-carbon electricity (%)']
    ].mean(numeric_only=True).reset_index()
    entity_summary_df.dropna(subset=['CO2 emissions (metric tons per capita)', 'Low-carbon electricity (%)'], inplace=True)

    fig_co2_low_carbon = px.scatter(
        entity_summary_df,
        x='Low-carbon electricity (%)',
        y='CO2 emissions (metric tons per capita)',
        hover_name='Entity',
        labels={
            'Low-carbon electricity (%)': 'Low-Carbon Electricity (%)',
            'CO2 emissions (metric tons per capita)': 'CO2 Emissions (metric tons/capita)'
        },
        log_y=True,
        trendline="ols"
    )
    fig_co2_low_carbon.update_yaxes(
        type="log",
        tickvals=[0.1, 1, 10, 100, 1000, 10000], # Adjusted tick values for better readability
        ticktext=['0.1', '1', '10', '100', '1k', '10k']
    )
    st.plotly_chart(fig_co2_low_carbon, use_container_width=True)

    st.markdown("---")
    st.subheader("Average CO2 Emissions per Capita by Continent (2019)")
    co2_continent_df = filtered_df[filtered_df['Year'] == 2019].copy()
    co2_continent_df.dropna(subset=['CO2 emissions (metric tons per capita)', 'Continent'], inplace=True)
    co2_continent_agg = co2_continent_df.groupby('Continent')['CO2 emissions (metric tons per capita)'].mean().reset_index()
    co2_continent_agg = co2_continent_agg.sort_values('CO2 emissions (metric tons per capita)', ascending=False)
    fig_co2_continent_bar = px.bar(
        co2_continent_agg,
        x='Continent',
        y='CO2 emissions (metric tons per capita)',
        color='Continent',
        labels={'CO2 emissions (metric tons per capita)': 'Avg. CO2 Emissions (metric tons/capita)'}
    )
    fig_co2_continent_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_co2_continent_bar, use_container_width=True)


    # --- Section 4 ---
    st.markdown("---")
    st.header("4. Identify Key Drivers 🤝")
    st.markdown("Beyond broad trends, what specific economic, demographic, and infrastructure factors correlate with progress in energy access and the shift to low-carbon solutions?")

    correlation_year = 2019
    correlation_df_base = filtered_df[filtered_df['Year'] == correlation_year].copy()

    columns_for_correlation_analysis = [
        'GDP per capita', 'CO2 emissions (metric tons per capita)',
        'Primary energy consumption (kWh/person)', 'Renewable energy share (%)',
        'Access to electricity (%)', 'Density (P/Km2)',
        'Low-carbon electricity (%)', 'Renewable-electricity-generating-capacity-per-capita',
        'Entity'
    ]
    correlation_df_base.dropna(subset=columns_for_correlation_analysis, inplace=True)
    correlation_df = correlation_df_base.groupby('Entity')[columns_for_correlation_analysis].mean(numeric_only=True).reset_index()

    plot_specs = [
        {
            'x_col': 'GDP per capita', 'y_col': 'Access to electricity (%)',
            'log_x': True, 'title': f'Electricity Access vs. GDP per Capita ({correlation_year})',
            'x_label': 'GDP per Capita (US$)', 'y_label': 'Access to Electricity (%)'
        },
        {
            'x_col': 'Density (P/Km2)', 'y_col': 'Access to electricity (%)',
            'log_x': True, 'title': f'Electricity Access vs. Population Density ({correlation_year})',
            'x_label': 'Population Density (P/Km²)', 'y_label': 'Access to Electricity (%)'
        },
        {
            'x_col': 'GDP per capita', 'y_col': 'Low-carbon electricity (%)',
            'log_x': True, 'title': f'Low-Carbon Electricity vs. GDP per Capita ({correlation_year})',
            'x_label': 'GDP per Capita (US$)', 'y_label': 'Low-Carbon Electricity (%)'
        },
        {
            'x_col': 'Renewable-electricity-generating-capacity-per-capita', 'y_col': 'Low-carbon electricity (%)',
            'log_x': True, 'title': f'Low-Carbon Electricity vs. Renewable Capacity per Capita ({correlation_year})',
            'x_label': 'Renewable Capacity per Capita (W/person)', 'y_label': 'Low-Carbon Electricity (%)'
        }
    ]

    num_cols = 2
    cols = st.columns(num_cols)
    col_idx = 0

    for spec in plot_specs:
        current_col = cols[col_idx % num_cols]
        with current_col:
            if spec['x_col'] in correlation_df.columns and spec['y_col'] in correlation_df.columns:
                fig = px.scatter(
                    correlation_df, x=spec['x_col'], y=spec['y_col'], hover_name='Entity',
                    title=spec['title'], labels={spec['x_col']: spec['x_label'], spec['y_col']: spec['y_label']},
                    log_x=spec.get('log_x', False), log_y=spec.get('log_y', False), trendline="ols"
                )
                fig.update_layout(
                    title_font_size=14, margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title_font_size=10, yaxis_title_font_size=10,
                    xaxis_tickfont_size=8, yaxis_tickfont_size=8
                )

                min_x_val = correlation_df[spec['x_col']].min()
                max_x_val = correlation_df[spec['x_col']].max()
                min_y_val = correlation_df[spec['y_col']].min() 
                max_y_val = correlation_df[spec['y_col']].max() 

                # Logic for X-axis range (remains as you had it)
                if spec.get('log_x', False):
                    if min_x_val <= 0: min_x_val = 0.001
                    x_range_min = np.log10(min_x_val * 0.9)
                    x_range_max = np.log10(max_x_val * 1.1)
                else:
                    buffer_percent = 0.05
                    range_span = max_x_val - min_x_val
                    x_range_min = min_x_val - (range_span * buffer_percent) 
                    x_range_max = max_x_val + (range_span * buffer_percent)
                fig.update_xaxes(range=[x_range_min, x_range_max])


                if spec.get('log_y', False):
                    if min_y_val <= 0: min_y_val = 0.001
                    y_range_min = np.log10(min_y_val * 0.9) 
                    y_range_max = np.log10(max_y_val * 1.1)
                else:
                    buffer_percent = 0.05
                    range_span_y = max_y_val - min_y_val
                    y_range_min = 0 
                    y_range_max = max_y_val + (range_span_y * buffer_percent) 

                fig.update_yaxes(range=[y_range_min, y_range_max])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Required columns for '{spec['title']}' not found in data or insufficient data after NaNs removed: {spec['x_col']}, {spec['y_col']}.")
        col_idx += 1


except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please ensure all necessary libraries are installed (`pip install -r requirements.txt`) and Kaggle API credentials are set up correctly.")


st.header("Summary of Key Elements 📝")
st.markdown("Most visuals used in this analysis were a combination of **bar and line charts**. This was intentional to clearly show trends over time or differences between categories. This dataset was well suited for these types of graphs in comparison to more statistical visuals such as box-whiskers or violin distributions. As it was a global dataset, the use of **chloropath maps** is a distinct feature.")
st.markdown("To add an exploratory component, I added the **correlation scatterplots with trendlines** (utalizing OLS linear regression). These dynamic plots help in visually identifying potential relationships between various energy, economic, and environmental indicators across contries.")
st.markdown("As with any analysis, this one contains a lot of data cleaning and preperations. Specifically, the *% of Renewable Energy Generation* was calculated based on the generation volumes in the provided datasets. Missing values were removed for purity of the data. To empower users to dynamically explore the data, there is a sidebar with selectable filters for year range and continents.")

st.markdown("---")
st.header("Final Evaluation ✅")
st.markdown("**Evaluation Process:** We conducted a Think-Aloud evaluation to gather qualitative feedback on the visualization's clarity and effectiveness. I presented the interactive dashboard to the participants, walked them through each section, and encouraged them to voice their thoughts, interpretations, and any points of confusion as they explored the charts.")
st.markdown("**Participants:** The evaluation involved three participants (my husband and 2 colleagues) with professional expertise in the renewable energy industry. Recruiting from this target audience provided valuable, domain-specific insights into the data's representation.")
st.markdown("**Results and Refinements:** The evaluation was highly positive and confirmed the report's utility. Key feedback included:")
st.markdown("* Energy generation is the best metric of energy development. Consumption or renewable energy share is more a metric for ability to convert or store efficiently.")
st.markdown("* Many of these metrics include other forms of energy -not just electricity. Renewable energy is typically only used for electricity.")
st.markdown("* Minor feedback on the correlation graphs going below 0% or above 100% with the trendline.")
st.markdown("* In section 2: Track Renewable Transition, they pointed out that it was difficult to see the breakdown between the energy generation mixes. I added a bar chart on the right to help with this.")

st.markdown("---")
st.header("Synthesis of Findings 🔍")
st.markdown("1. **Understanding Global Energy Access:**")
st.markdown("* Access to electricity and clean cooking fuels has significantly increased between 2000-2019.")
st.markdown("* Electricity access and clean fuels access is correlated with Africa having the lowest access.")

st.markdown("2. **Track Renewable Transition:**")
st.markdown("* Renewable electricity generation has been increasing overtime along with fossil fuel electricity generation.")
st.markdown("* Renewable electricity generation has increased across each continent with Europe experiencing the most growth.")
st.markdown("* Surprisingly, China produces the most electricity from renewables as an absolute value! Though it only contributes to 14.45% of its total electricity generation.")
st.markdown("* Central African countries' electricity is mostly from renewable energy.")

st.markdown("3. **Evaluate Environmental Impact:**")
st.markdown("* The relationship between low-carbon electricity share and CO₂ emissions per capita is not a simple negative correlation. This surprising result suggests that other factors, such as industrial output, transportation emissions, and total energy consumption, are significant contributors to a country's carbon footprint, and that decarbonizing electricity is only one piece of a larger puzzle.")

st.markdown("4. **Identify Key Drivers:**")
st.markdown("* Most of these required a log scale graph to make sense of the relationships. But none of the correlations seem surprising.")
st.markdown("* These correlations make me want to explore solutions to decreasing our dependence on fossil fuels in exchange for sustainable sources.")

st.markdown("**Future Iterations:**")
st.markdown("* I liked the use of line, bar, and scatterplots. If I were to refine this again, I would want to find a use case for more statistical graphs or forecasts.")
st.markdown("* I would want to pull in more data sources to explore more features such as weather data, political policies, electricity usage, etc.")


st.markdown("---")
st.header("Sources 📚")
st.markdown("Energy Source: https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy/data")
st.markdown("Country Source: https://www.kaggle.com/datasets/andradaolteanu/country-mapping-iso-continent-region")
st.markdown("World Bank, International Energy Agency, and ourworldindata.org")
st.markdown("The data is gathered from websites including World Bank and International Energy Agency. The majority of it is collected from this site https://ourworldindata.org/sdgs/affordable-clean-energy. Some of the columns/features are taken from other top Kaggle datasets.")

st.markdown("---") # Add a separator before the footer
st.markdown("© 2025 Holly Nereson. All rights reserved.")
st.markdown("Built for **CU Boulder's DTSA 5304: Fundamentals of Data Visualization** course.")