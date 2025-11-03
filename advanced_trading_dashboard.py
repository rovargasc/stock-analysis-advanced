import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from portfolio_analyzer import PortfolioAnalyzer, StockAnalysis
import time
from datetime import datetime, timedelta
from typing import Dict, List
import json

# Importar las funciones originales del dashboard
from trading_dashboard import (
    get_api_key, format_number, format_currency, 
    HybridAPI, EnhancedTechnicalAnalysis, create_enhanced_chart
)

def create_sector_analysis_chart(analyses: List[StockAnalysis]) -> go.Figure:
    """Crear gráfico de análisis por sector"""
    sector_data = {}
    
    for analysis in analyses:
        sector = analysis.sector
        if sector not in sector_data:
            sector_data[sector] = {
                'count': 0,
                'avg_score': 0,
                'total_market_cap': 0,
                'stocks': []
            }
        
        sector_data[sector]['count'] += 1
        sector_data[sector]['avg_score'] += analysis.overall_score
        sector_data[sector]['total_market_cap'] += analysis.market_cap if analysis.market_cap > 0 else 0
        sector_data[sector]['stocks'].append(analysis.symbol)
    
    # Calcular promedios
    sectors = []
    avg_scores = []
    counts = []
    market_caps = []
    
    for sector, data in sector_data.items():
        if data['count'] > 0:
            sectors.append(sector)
            avg_scores.append(data['avg_score'] / data['count'])
            counts.append(data['count'])
            market_caps.append(data['total_market_cap'] / 1000)  # En billones
    
    # Crear gráfico de burbujas
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=avg_scores,
        y=counts,
        mode='markers+text',
        marker=dict(
            size=[min(50, max(10, mc/10)) for mc in market_caps],  # Tamaño basado en market cap
            color=avg_scores,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Score Promedio"),
            line=dict(width=1, color='black')
        ),
        text=sectors,
        textposition="middle center",
        textfont=dict(size=10, color='white'),
        hovertemplate='<b>%{text}</b><br>' +
                      'Score Promedio: %{x:.1f}<br>' +
                      'Número de Acciones: %{y}<br>' +
                      'Market Cap Total: $%{customdata:.1f}B<br>' +
                      '<extra></extra>',
        customdata=market_caps
    ))
    
    fig.update_layout(
        title='Análisis por Sector (Tamaño = Market Cap Total)',
        xaxis_title='Score Promedio',
        yaxis_title='Número de Acciones Analizadas',
        height=600,
        showlegend=False
    )
    
    return fig

def create_risk_return_chart(analyses: List[StockAnalysis]) -> go.Figure:
    """Crear gráfico riesgo vs retorno"""
    returns = []
    risks = []
    symbols = []
    scores = []
    colors = []
    
    for analysis in analyses:
        if isinstance(analysis.upside_potential, (int, float)):
            returns.append(analysis.upside_potential)
            
            # Mapear riesgo a valor numérico
            risk_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            risks.append(risk_map.get(analysis.risk_level, 2))
            
            symbols.append(analysis.symbol)
            scores.append(analysis.overall_score)
            
            # Color basado en score
            if analysis.overall_score >= 70:
                colors.append('green')
            elif analysis.overall_score >= 50:
                colors.append('orange')
            else:
                colors.append('red')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=risks,
        y=returns,
        mode='markers+text',
        marker=dict(
            size=12,
            color=scores,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Overall Score"),
            line=dict(width=1, color='black')
        ),
        text=symbols,
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>' +
                      'Upside Potential: %{y:.1f}%<br>' +
                      'Risk Level: %{customdata}<br>' +
                      'Overall Score: %{marker.color:.1f}<br>' +
                      '<extra></extra>',
        customdata=[list(risk_map.keys())[r-1] for r in risks]
    ))
    
    # Añadir líneas de referencia
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Sin upside")
    fig.add_vline(x=2, line_dash="dash", line_color="gray", annotation_text="Riesgo Medio")
    
    fig.update_layout(
        title='Riesgo vs Potencial de Retorno',
        xaxis_title='Nivel de Riesgo',
        yaxis_title='Potencial de Upside (%)',
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Bajo', 'Medio', 'Alto']
        ),
        height=600
    )
    
    return fig

def create_score_comparison_chart(analyses: List[StockAnalysis], top_n: int = 20) -> go.Figure:
    """Crear gráfico de comparación de scores"""
    # Tomar top N acciones por score general
    top_analyses = sorted(analyses, key=lambda x: x.overall_score, reverse=True)[:top_n]
    
    symbols = [a.symbol for a in top_analyses]
    fundamental_scores = [a.fundamental_score for a in top_analyses]
    technical_scores = [a.technical_score for a in top_analyses]
    sentiment_scores = [a.sentiment_score for a in top_analyses]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Fundamental',
        x=symbols,
        y=fundamental_scores,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Technical',
        x=symbols,
        y=technical_scores,
        marker_color='lightgreen'
    ))
    
    fig.add_trace(go.Bar(
        name='Sentiment',
        x=symbols,
        y=sentiment_scores,
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title=f'Comparación de Scores - Top {top_n} Acciones',
        xaxis_title='Símbolos',
        yaxis_title='Score (0-100)',
        barmode='group',
        height=600,
        showlegend=True
    )
    
    return fig

def create_metrics_heatmap(analyses: List[StockAnalysis]) -> go.Figure:
    """Crear heatmap de métricas financieras"""
    # Seleccionar top 20 acciones
    top_analyses = sorted(analyses, key=lambda x: x.overall_score, reverse=True)[:20]
    
    symbols = [a.symbol for a in top_analyses]
    
    # Normalizar métricas para el heatmap (0-100)
    metrics_data = []
    metric_names = ['P/E Ratio', 'P/B Ratio', 'ROE', 'ROA', 'Profit Margin', 'Revenue Growth']
    
    for analysis in top_analyses:
        row = []
        
        # P/E (invertir - menor es mejor, normalizar)
        pe = analysis.pe_ratio if isinstance(analysis.pe_ratio, (int, float)) and analysis.pe_ratio > 0 else 30
        pe_score = max(0, 100 - (pe - 10) * 2) if pe > 0 else 50
        row.append(min(100, max(0, pe_score)))
        
        # P/B (invertir - menor es mejor)
        pb = analysis.pb_ratio if isinstance(analysis.pb_ratio, (int, float)) and analysis.pb_ratio > 0 else 3
        pb_score = max(0, 100 - (pb - 1) * 20) if pb > 0 else 50
        row.append(min(100, max(0, pb_score)))
        
        # ROE (mayor es mejor)
        roe = analysis.roe if isinstance(analysis.roe, (int, float)) else 0
        roe_score = min(100, max(0, roe * 4)) if roe >= 0 else 0
        row.append(roe_score)
        
        # ROA (mayor es mejor)
        roa = analysis.roa if isinstance(analysis.roa, (int, float)) else 0
        roa_score = min(100, max(0, roa * 8)) if roa >= 0 else 0
        row.append(roa_score)
        
        # Profit Margin (mayor es mejor)
        pm = analysis.profit_margin if isinstance(analysis.profit_margin, (int, float)) else 0
        pm_score = min(100, max(0, pm * 4)) if pm >= 0 else 0
        row.append(pm_score)
        
        # Revenue Growth (mayor es mejor, pero cap en 100)
        rg = analysis.revenue_growth if isinstance(analysis.revenue_growth, (int, float)) else 0
        rg_score = min(100, max(0, (rg + 10) * 2)) if rg > -50 else 0
        row.append(rg_score)
        
        metrics_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=metrics_data,
        x=metric_names,
        y=symbols,
        colorscale='RdYlGn',
        showscale=True,
        colorbar=dict(title="Score Normalizado"),
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>' +
                      '%{x}: %{z:.1f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Heatmap de Métricas Financieras (Top 20)',
        height=600,
        xaxis_title='Métricas',
        yaxis_title='Símbolos'
    )
    
    return fig

def display_stock_cards(analyses: List[StockAnalysis], title: str):
    """Mostrar tarjetas de acciones"""
    st.subheader(title)
    
    cols = st.columns(3)
    for i, analysis in enumerate(analyses):
        with cols[i % 3]:
            # Determinar color del score
            if analysis.overall_score >= 70:
                score_color = "🟢"
            elif analysis.overall_score >= 50:
                score_color = "🟡"
            else:
                score_color = "🔴"
            
            # Determinar emoji del consenso
            consensus_emoji = {
                "BUY": "🚀",
                "HOLD": "⏸️",
                "SELL": "⬇️"
            }
            
            upside_text = f"{analysis.upside_potential:.1f}%" if isinstance(analysis.upside_potential, (int, float)) else "N/A"
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; 
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
                <h4>{analysis.symbol} - {analysis.name[:25]}...</h4>
                <p><strong>Sector:</strong> {analysis.sector}</p>
                <p><strong>Precio:</strong> ${analysis.price:.2f}</p>
                <p><strong>Score:</strong> {score_color} {analysis.overall_score:.1f}/100</p>
                <p><strong>Consenso:</strong> {consensus_emoji.get(analysis.analyst_consensus, "⚪")} {analysis.analyst_consensus}</p>
                <p><strong>Upside:</strong> {upside_text}</p>
                <p><strong>Riesgo:</strong> {analysis.risk_level}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Advanced Trading Dashboard", 
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Advanced Trading Dashboard")
    st.markdown("*Análisis inteligente de carteras de inversión con screening automático*")
    
    # CSS personalizado
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .top-performer {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .risk-card {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Obtener API key
    try:
        api_key = get_api_key()
        if not api_key:
            st.error("🚫 No se pudo obtener la API key")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error obteniendo API key: {e}")
        st.stop()
    
    # Inicializar analizador
    try:
        analyzer = PortfolioAnalyzer(api_key)
        st.success("✅ Sistema de análisis de cartera inicializado")
    except Exception as e:
        st.error(f"❌ Error inicializando: {e}")
        return
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración de Análisis")
        
        # Selección de universo de acciones
        st.subheader("📈 Universo de Acciones")
        available_categories = list(analyzer.stock_universe.keys())
        selected_categories = st.multiselect(
            "Seleccionar categorías:",
            available_categories,
            default=["Large Cap Tech", "Financial", "Healthcare"]
        )
        
        # Límite de acciones por categoría
        stocks_per_category = st.slider("Acciones por categoría", 3, 10, 5)
        
        # Criterios de filtrado
        st.subheader("🔍 Filtros")
        min_market_cap = st.selectbox(
            "Market Cap mínimo",
            ["Sin límite", "> $1B", "> $10B", "> $50B"],
            index=1
        )
        
        max_pe_ratio = st.slider("P/E máximo", 0, 100, 50)
        min_roe = st.slider("ROE mínimo (%)", -50, 50, 5)
        
        risk_tolerance = st.selectbox(
            "Tolerancia al riesgo",
            ["Todos", "Solo Bajo", "Bajo y Medio", "Solo Alto"],
            index=2
        )
        
        # Pesos personalizados para scoring
        st.subheader("⚖️ Pesos de Análisis")
        fundamental_weight = st.slider("Peso Fundamental", 0.0, 1.0, 0.4, 0.1)
        technical_weight = st.slider("Peso Técnico", 0.0, 1.0, 0.3, 0.1)
        sentiment_weight = 1.0 - fundamental_weight - technical_weight
        
        if sentiment_weight < 0:
            st.error("Los pesos deben sumar máximo 1.0")
        else:
            st.info(f"Peso Sentimiento: {sentiment_weight:.1f}")
            analyzer.scoring_weights = {
                'fundamental': fundamental_weight,
                'technical': technical_weight,
                'sentiment': sentiment_weight
            }
        
        # Botón de análisis
        if st.button("🚀 Ejecutar Análisis Completo", type="primary"):
            st.session_state.run_analysis = True
    
    # Ejecutar análisis
    if st.sidebar.button("🔄 Ejecutar Análisis Rápido") or st.session_state.get('run_analysis', False):
        if not selected_categories:
            st.error("⚠️ Selecciona al menos una categoría de acciones")
            return
        
        # Preparar lista de símbolos
        symbols_to_analyze = []
        for category in selected_categories:
            symbols_to_analyze.extend(analyzer.stock_universe[category][:stocks_per_category])
        
        st.info(f"🔍 Analizando {len(symbols_to_analyze)} acciones...")
        
        # Obtener datos en paralelo
        with st.spinner("Obteniendo datos de mercado..."):
            start_time = time.time()
            stock_data = analyzer.get_stock_data_parallel(symbols_to_analyze, max_workers=8)
        
        if not stock_data:
            st.error("❌ No se pudieron obtener datos de ninguna acción")
            return
        
        # Analizar cada acción
        analyses = []
        with st.spinner("Ejecutando análisis..."):
            for symbol, data in stock_data.items():
                analysis = analyzer.analyze_stock(symbol, data)
                if analysis:
                    analyses.append(analysis)
        
        # Aplicar filtros
        filtered_analyses = []
        for analysis in analyses:
            # Filtro de market cap
            if min_market_cap != "Sin límite":
                min_cap = {"> $1B": 1000, "> $10B": 10000, "> $50B": 50000}[min_market_cap]
                if analysis.market_cap < min_cap:
                    continue
            
            # Filtro P/E
            if isinstance(analysis.pe_ratio, (int, float)) and analysis.pe_ratio > max_pe_ratio:
                continue
            
            # Filtro ROE
            if isinstance(analysis.roe, (int, float)) and analysis.roe < min_roe:
                continue
            
            # Filtro de riesgo
            if risk_tolerance == "Solo Bajo" and analysis.risk_level != "LOW":
                continue
            elif risk_tolerance == "Bajo y Medio" and analysis.risk_level == "HIGH":
                continue
            elif risk_tolerance == "Solo Alto" and analysis.risk_level != "HIGH":
                continue
            
            filtered_analyses.append(analysis)
        
        if not filtered_analyses:
            st.error("❌ No se encontraron acciones que cumplan los criterios")
            return
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        st.success(f"✅ Análisis completado en {analysis_time:.1f} segundos")
        st.info(f"📊 Se analizaron {len(filtered_analyses)} acciones de {len(symbols_to_analyze)} totales")
        
        # Almacenar en session state
        st.session_state.analyses = filtered_analyses
        st.session_state.run_analysis = False
    
    # Mostrar resultados si existen
    if 'analyses' in st.session_state and st.session_state.analyses:
        analyses = st.session_state.analyses
        
        # Métricas principales
        st.header("📈 Resumen Ejecutivo")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_score = np.mean([a.overall_score for a in analyses])
            st.metric("Score Promedio", f"{avg_score:.1f}")
        
        with col2:
            top_score = max([a.overall_score for a in analyses])
            best_stock = next(a for a in analyses if a.overall_score == top_score)
            st.metric("Mejor Acción", best_stock.symbol, f"{top_score:.1f}")
        
        with col3:
            buy_recommendations = len([a for a in analyses if a.analyst_consensus == "BUY"])
            st.metric("Recomendaciones BUY", buy_recommendations)
        
        with col4:
            high_upside = len([a for a in analyses if isinstance(a.upside_potential, (int, float)) and a.upside_potential > 20])
            st.metric("Alto Potencial (>20%)", high_upside)
        
        with col5:
            low_risk = len([a for a in analyses if a.risk_level == "LOW"])
            st.metric("Bajo Riesgo", low_risk)
        
        # Pestañas principales
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Rankings", "📊 Análisis Visual", "🔍 Detalles", "📈 Comparación", "💼 Cartera Sugerida"
        ])
        
        with tab1:
            st.header("🏆 Rankings de Acciones")
            
            # Rankings por diferentes criterios
            ranking_options = {
                "🎯 Mejor Score General": "overall_score",
                "📈 Mayor Potencial de Upside": "upside_potential", 
                "💪 Mejores Fundamentales": "fundamental_score",
                "📊 Mejores Técnicos": "technical_score",
                "🛡️ Menor Riesgo": "low_risk",
                "🚀 Crecimiento": "growth",
                "💰 Valor": "value"
            }
            
            selected_ranking = st.selectbox("Seleccionar criterio de ranking:", list(ranking_options.keys()))
            top_n = st.slider("Mostrar top N acciones:", 5, 20, 10)
            
            criterion = ranking_options[selected_ranking]
            top_stocks = analyzer.get_top_stocks_by_criteria(analyses, criterion, top_n)
            
            # Mostrar tarjetas
            display_stock_cards(top_stocks, selected_ranking)
            
            # Tabla detallada
            st.subheader("📋 Tabla Detallada")
            summary_df = analyzer.create_portfolio_summary(top_stocks)
            st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.header("📊 Análisis Visual")
            
            # Gráfico de sectores
            sector_chart = create_sector_analysis_chart(analyses)
            st.plotly_chart(sector_chart, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico riesgo-retorno
                risk_return_chart = create_risk_return_chart(analyses)
                st.plotly_chart(risk_return_chart, use_container_width=True)
            
            with col2:
                # Comparación de scores
                score_chart = create_score_comparison_chart(analyses)
                st.plotly_chart(score_chart, use_container_width=True)
            
            # Heatmap de métricas
            heatmap_chart = create_metrics_heatmap(analyses)
            st.plotly_chart(heatmap_chart, use_container_width=True)
        
        with tab3:
            st.header("🔍 Análisis Detallado")
            
            # Selector de acción para análisis individual
            stock_symbols = [a.symbol for a in analyses]
            selected_stock_symbol = st.selectbox("Seleccionar acción para análisis detallado:", stock_symbols)
            
            selected_analysis = next(a for a in analyses if a.symbol == selected_stock_symbol)
            
            # Mostrar análisis detallado (reutilizar código del dashboard original)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("💰 Información Básica")
                st.write(f"**Nombre:** {selected_analysis.name}")
                st.write(f"**Sector:** {selected_analysis.sector}")
                st.write(f"**Precio:** ${selected_analysis.price:.2f}")
                if selected_analysis.market_cap > 0:
                    st.write(f"**Market Cap:** ${selected_analysis.market_cap/1000:.1f}B")
                st.write(f"**Nivel de Riesgo:** {selected_analysis.risk_level}")
            
            with col2:
                st.subheader("📊 Métricas Financieras")
                if isinstance(selected_analysis.pe_ratio, (int, float)) and selected_analysis.pe_ratio > 0:
                    st.write(f"**P/E Ratio:** {selected_analysis.pe_ratio:.1f}")
                if isinstance(selected_analysis.pb_ratio, (int, float)) and selected_analysis.pb_ratio > 0:
                    st.write(f"**P/B Ratio:** {selected_analysis.pb_ratio:.1f}")
                if isinstance(selected_analysis.roe, (int, float)):
                    st.write(f"**ROE:** {selected_analysis.roe:.1f}%")
                if isinstance(selected_analysis.roa, (int, float)):
                    st.write(f"**ROA:** {selected_analysis.roa:.1f}%")
                if isinstance(selected_analysis.revenue_growth, (int, float)):
                    st.write(f"**Crecimiento Ingresos:** {selected_analysis.revenue_growth:.1f}%")
            
            with col3:
                st.subheader("🎯 Análisis y Recomendaciones")
                st.write(f"**RSI:** {selected_analysis.rsi:.1f}")
                st.write(f"**MACD:** {selected_analysis.macd_signal}")
                st.write(f"**Consenso Analistas:** {selected_analysis.analyst_consensus}")
                if selected_analysis.target_price > 0:
                    st.write(f"**Precio Objetivo:** ${selected_analysis.target_price:.2f}")
                if isinstance(selected_analysis.upside_potential, (int, float)):
                    st.write(f"**Potencial Upside:** {selected_analysis.upside_potential:.1f}%")
            
            # Scores
            st.subheader("📈 Scores de Análisis")
            score_col1, score_col2, score_col3, score_col4 = st.columns(4)
            
            with score_col1:
                st.metric("Overall", f"{selected_analysis.overall_score:.1f}/100")
            with score_col2:
                st.metric("Fundamental", f"{selected_analysis.fundamental_score:.1f}/100")
            with score_col3:
                st.metric("Técnico", f"{selected_analysis.technical_score:.1f}/100")
            with score_col4:
                st.metric("Sentimiento", f"{selected_analysis.sentiment_score:.1f}/100")
        
        with tab4:
            st.header("📈 Comparación Avanzada")
            
            # Selector múltiple para comparar acciones
            symbols_to_compare = st.multiselect(
                "Seleccionar acciones para comparar:",
                [a.symbol for a in analyses],
                default=[a.symbol for a in analyses[:5]]
            )
            
            if len(symbols_to_compare) >= 2:
                comparison_analyses = [a for a in analyses if a.symbol in symbols_to_compare]
                
                # Tabla de comparación
                comparison_data = []
                for analysis in comparison_analyses:
                    comparison_data.append({
                        'Symbol': analysis.symbol,
                        'Price': f"${analysis.price:.2f}",
                        'Overall Score': analysis.overall_score,
                        'Fundamental': analysis.fundamental_score,
                        'Technical': analysis.technical_score,
                        'Sentiment': analysis.sentiment_score,
                        'P/E': analysis.pe_ratio if isinstance(analysis.pe_ratio, (int, float)) else "N/A",
                        'ROE': f"{analysis.roe:.1f}%" if isinstance(analysis.roe, (int, float)) else "N/A",
                        'Upside': f"{analysis.upside_potential:.1f}%" if isinstance(analysis.upside_potential, (int, float)) else "N/A",
                        'Risk': analysis.risk_level,
                        'Consensus': analysis.analyst_consensus
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Gráfico de radar para comparación
                categories = ['Overall Score', 'Fundamental', 'Technical', 'Sentiment']
                
                fig = go.Figure()
                
                for analysis in comparison_analyses:
                    fig.add_trace(go.Scatterpolar(
                        r=[analysis.overall_score, analysis.fundamental_score, 
                           analysis.technical_score, analysis.sentiment_score],
                        theta=categories,
                        fill='toself',
                        name=analysis.symbol
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Comparación de Scores (Gráfico Radar)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.header("💼 Cartera Sugerida")
            
            # Configuración de cartera
            col1, col2 = st.columns(2)
            
            with col1:
                portfolio_size = st.slider("Número de acciones en cartera:", 3, 15, 8)
                portfolio_strategy = st.selectbox(
                    "Estrategia de cartera:",
                    ["Balanceada", "Crecimiento", "Valor", "Bajo Riesgo", "Alto Potencial"]
                )
            
            with col2:
                max_risk_exposure = st.slider("% máximo en acciones de alto riesgo:", 0, 50, 20)
                min_sectors = st.slider("Número mínimo de sectores:", 2, 6, 3)
            
            # Generar cartera sugerida
            if st.button("🎯 Generar Cartera Sugerida"):
                # Lógica de selección basada en estrategia
                if portfolio_strategy == "Balanceada":
                    # Mix de todas las categorías
                    suggested_stocks = sorted(analyses, key=lambda x: x.overall_score, reverse=True)[:portfolio_size]
                elif portfolio_strategy == "Crecimiento":
                    # Priorizar crecimiento de ingresos y upside potential
                    growth_stocks = [a for a in analyses if isinstance(a.revenue_growth, (int, float)) and a.revenue_growth > 5]
                    suggested_stocks = sorted(growth_stocks, key=lambda x: (x.revenue_growth, x.overall_score), reverse=True)[:portfolio_size]
                elif portfolio_strategy == "Valor":
                    # Priorizar métricas de valor
                    value_stocks = [a for a in analyses if isinstance(a.pe_ratio, (int, float)) and a.pe_ratio > 0 and a.pe_ratio < 25]
                    suggested_stocks = sorted(value_stocks, key=lambda x: (x.fundamental_score, -x.pe_ratio), reverse=True)[:portfolio_size]
                elif portfolio_strategy == "Bajo Riesgo":
                    # Solo acciones de bajo riesgo con buenos fundamentales
                    low_risk_stocks = [a for a in analyses if a.risk_level == "LOW"]
                    suggested_stocks = sorted(low_risk_stocks, key=lambda x: x.fundamental_score, reverse=True)[:portfolio_size]
                else:  # Alto Potencial
                    # Priorizar upside potential
                    high_potential = [a for a in analyses if isinstance(a.upside_potential, (int, float)) and a.upside_potential > 10]
                    suggested_stocks = sorted(high_potential, key=lambda x: x.upside_potential, reverse=True)[:portfolio_size]
                
                # Verificar diversificación por sector
                sector_count = len(set(s.sector for s in suggested_stocks))
                if sector_count < min_sectors:
                    st.warning(f"⚠️ La cartera sugerida tiene solo {sector_count} sectores. Considera ajustar los criterios.")
                
                # Verificar exposición al riesgo
                high_risk_count = len([s for s in suggested_stocks if s.risk_level == "HIGH"])
                risk_exposure = (high_risk_count / len(suggested_stocks)) * 100
                
                if risk_exposure > max_risk_exposure:
                    st.warning(f"⚠️ Exposición al riesgo alto: {risk_exposure:.1f}% (límite: {max_risk_exposure}%)")
                
                # Mostrar cartera sugerida
                st.subheader("🎯 Tu Cartera Sugerida")
                
                total_score = np.mean([s.overall_score for s in suggested_stocks])
                total_upside = np.mean([s.upside_potential for s in suggested_stocks if isinstance(s.upside_potential, (int, float))])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score Promedio Cartera", f"{total_score:.1f}")
                with col2:
                    st.metric("Upside Promedio", f"{total_upside:.1f}%" if not np.isnan(total_upside) else "N/A")
                with col3:
                    st.metric("Diversificación", f"{sector_count} sectores")
                
                # Tabla de la cartera con pesos sugeridos
                portfolio_data = []
                equal_weight = 100 / len(suggested_stocks)
                
                for stock in suggested_stocks:
                    portfolio_data.append({
                        'Symbol': stock.symbol,
                        'Name': stock.name[:25] + "..." if len(stock.name) > 25 else stock.name,
                        'Sector': stock.sector,
                        'Peso Sugerido (%)': f"{equal_weight:.1f}",
                        'Price': f"${stock.price:.2f}",
                        'Score': f"{stock.overall_score:.1f}",
                        'Upside': f"{stock.upside_potential:.1f}%" if isinstance(stock.upside_potential, (int, float)) else "N/A",
                        'Risk': stock.risk_level,
                        'Consensus': stock.analyst_consensus
                    })
                
                portfolio_df = pd.DataFrame(portfolio_data)
                st.dataframe(portfolio_df, use_container_width=True)
                
                # Gráfico de sectores en la cartera
                sector_weights = {}
                for stock in suggested_stocks:
                    sector = stock.sector
                    if sector in sector_weights:
                        sector_weights[sector] += equal_weight
                    else:
                        sector_weights[sector] = equal_weight
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(sector_weights.keys()),
                    values=list(sector_weights.values()),
                    hole=0.3
                )])
                
                fig_pie.update_layout(
                    title="Distribución por Sector en la Cartera",
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Resumen de riesgos
                st.subheader("⚠️ Análisis de Riesgos")
                risk_summary = {
                    "LOW": len([s for s in suggested_stocks if s.risk_level == "LOW"]),
                    "MEDIUM": len([s for s in suggested_stocks if s.risk_level == "MEDIUM"]),
                    "HIGH": len([s for s in suggested_stocks if s.risk_level == "HIGH"])
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🟢 Riesgo Bajo", risk_summary["LOW"])
                with col2:
                    st.metric("🟡 Riesgo Medio", risk_summary["MEDIUM"])
                with col3:
                    st.metric("🔴 Riesgo Alto", risk_summary["HIGH"])
    
    else:
        # Página de inicio si no hay análisis
        st.header("🚀 Bienvenido al Advanced Trading Dashboard")
        
        st.markdown("""
        Este dashboard te permite:
        
        * 📊 **Analizar múltiples acciones simultáneamente**
        * 🏆 **Generar rankings inteligentes** basados en análisis fundamental, técnico y de sentimiento
        * 📈 **Visualizar comparaciones** entre sectores, riesgos y potencial de retorno
        * 💼 **Crear carteras sugeridas** basadas en diferentes estrategias
        * 🔍 **Filtrar acciones** según tus criterios específicos
        
        **Para comenzar:**
        1. Configura tus preferencias en el panel lateral
        2. Selecciona las categorías de acciones que te interesan
        3. Haz clic en "Ejecutar Análisis Completo"
        
        **Características avanzadas:**
        - Análisis en paralelo para mayor velocidad
        - Sistema de scoring multicriteria
        - Filtros personalizables
        - Visualizaciones interactivas
        - Recomendaciones de cartera automatizadas
        """)
        
        # Mostrar ejemplo de categorías disponibles
        st.subheader("📋 Categorías de Acciones Disponibles")
        for category, stocks in analyzer.stock_universe.items():
            with st.expander(f"{category} ({len(stocks)} acciones)"):
                st.write(", ".join(stocks))

if __name__ == "__main__":
    main()