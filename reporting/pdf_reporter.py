"""Generate a self-contained, printable ECG analysis report in memory."""
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

from reporting.schemas import ECGStressReport, ModelComparisonResult


class PDFReportGenerator:
    @staticmethod
    def generate(report: ECGStressReport) -> bytes:
        stream = BytesIO()
        ink = colors.HexColor('#24312b')
        muted = colors.HexColor('#59685e')
        line = colors.HexColor('#d4ddd6')
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle('ReportTitle', fontName='Helvetica-Bold', fontSize=23, leading=27, textColor=ink, spaceAfter=10))
        styles.add(ParagraphStyle('SectionTitle', fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=ink, spaceBefore=16, spaceAfter=9))
        styles.add(ParagraphStyle('ReportBody', fontName='Helvetica', fontSize=10, leading=14, textColor=ink, spaceAfter=7))
        styles.add(ParagraphStyle('ReportSmall', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=muted, spaceAfter=5))
        width = A4[0] - 36 * mm
        doc = SimpleDocTemplate(stream, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                                topMargin=18*mm, bottomMargin=19*mm,
                                title='ECG Analysis - ' + report.report_id, author='ECG research prototype')

        def text(value, style='ReportBody'):
            return Paragraph(escape(str(value)), styles[style])

        def table(rows, widths, font_size=9):
            cells = [[text(value, 'ReportSmall') for value in row] for row in rows]
            item = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
            item.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#edf2ee')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LINEBELOW', (0,0), (-1,0), .7, line),
                ('LINEBELOW', (0,1), (-1,-1), .35, line),
                ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ]))
            return item

        metadata = report.pipeline_metadata
        models = report.model_comparison or [ModelComparisonResult(
            model_id=metadata.get('model_id', 'selected'), model_name=metadata.get('model_name', metadata.get('model_used', 'Selected model')),
            prediction=report.prediction, prediction_time_ms=metadata.get('prediction_time_ms', 0),
            explanation_time_ms=metadata.get('explanation_time_ms', 0), explanation_method=metadata.get('explanation_method'),
            included_training=metadata.get('included_training', False),
        )]
        is_demo = any(item.prediction and item.prediction.is_demo for item in models)
        story = [text('ECG analysis report', 'ReportTitle'), text(report.source_file or 'Recording', 'SectionTitle'),
                 text(f'Report {report.report_id} | Generated {report.timestamp}', 'ReportSmall')]
        if is_demo:
            story.append(text('DEMONSTRATION: Synthetic teaching models are used. Predictions, probabilities and SHAP contributions are illustrative, not validated stress assessments.'))
        else:
            story.append(text('Locally trained model output. Model confidence is not measured accuracy or an independent clinical assessment.'))
        duration = metadata.get('duration_s', metadata.get('duration_sec', 'Not recorded'))
        story.append(table([
            ['Recording duration (s)', 'Sampling rate (Hz)', 'Detected R-peaks'],
            [duration, metadata.get('sampling_rate_hz', 'Not recorded'), metadata.get('r_peaks_detected', 'Not recorded')],
        ], [width/3]*3))
        story.append(text('HRV measurements', 'SectionTitle'))
        names = {'SDNN':'SDNN', 'RMSSD':'RMSSD', 'LF_HF_Ratio':'LF / HF ratio', 'SampEn':'Sample entropy', 'Sample_Entropy':'Sample entropy'}
        rows = [['Measurement', 'Value', 'Unit']]
        for key, value in report.key_hrv_summary.items():
            rows.append([names.get(key, key), 'Unavailable' if value is None else f'{value:.3f}', 'ms' if key in ('SDNN','RMSSD','MeanRR') else 'Unitless'])
        story.append(table(rows, [width*.47, width*.31, width*.22]))
        story.append(text('Model comparison' if report.model_comparison else 'Selected model', 'SectionTitle'))
        rows = [['Model', 'Predicted state', 'Stress probability', 'Confidence', 'Predict ms', 'SHAP ms']]
        for item in models:
            prediction = item.prediction
            rows.append([item.model_name, prediction.prediction_label if prediction else 'Unavailable',
                         f'{prediction.stress_probability:.1%}' if prediction else '-',
                         f'{prediction.confidence_score:.1%}' if prediction else '-',
                         f'{item.prediction_time_ms:.1f}' if prediction else '-', f'{item.explanation_time_ms:.1f}' if prediction else '-'])
        story.append(table(rows, [width*.24, width*.18, width*.17, width*.14, width*.13, width*.14]))
        story.append(Spacer(1, 9))
        story.append(text('All models use the same extracted HRV measurements. Confidence is relative to the predicted class; stress probability always refers to Stress. These values do not rank model accuracy.', 'ReportSmall'))
        story.append(text('Timings include demo fitting on a model\'s first use, and SHAP setup/computation. Warm up models before formal speed comparisons.', 'ReportSmall'))
        for item in models:
            if item.error:
                story.append(text(item.model_name + ': ' + item.error, 'ReportSmall'))
        for warning in metadata.get('warnings', []):
            story.append(text('Measurement note: ' + warning, 'ReportSmall'))
        successful = [item for item in models if item.prediction]
        for index, item in enumerate(successful):
            if index % 2 == 0:
                story.append(PageBreak())
                story.append(text('Feature explanations', 'ReportTitle'))
                story.append(text('SHAP values describe contributions to the model\'s Stress probability. Positive values move toward Stress; negative values move toward No-Stress. They are not proof of a physiological cause.', 'ReportSmall'))
            heading = f'{item.model_name} - {item.prediction.prediction_label}'
            rows = [['Feature', 'Measured / input value', 'SHAP value', 'Direction']]
            for feature in item.prediction.top_shap_features:
                rows.append([names.get(feature.feature_name, feature.feature_name), f'{feature.feature_value:.3f}',
                             f'{feature.shap_value:+.4f}', 'Toward Stress' if feature.shap_value > 0 else 'Toward No-Stress' if feature.shap_value < 0 else 'Neutral'])
            story.append(KeepTogether([
                text(heading, 'SectionTitle'), text(f'{item.explanation_method or "SHAP"} | Stress probability {item.prediction.stress_probability:.1%} | ' + ('Demo model' if item.prediction.is_demo else 'Locally trained model'), 'ReportSmall'),
                table(rows, [width*.25, width*.25, width*.20, width*.30]), Spacer(1,12),
            ]))
        def footer(canvas, document):
            canvas.saveState()
            canvas.setStrokeColor(line); canvas.line(18*mm, 14*mm, A4[0]-18*mm, 14*mm)
            canvas.setFont('Helvetica', 8); canvas.setFillColor(muted)
            canvas.drawString(18*mm, 10*mm, 'ECG / HRV research prototype')
            canvas.drawRightString(A4[0]-18*mm, 10*mm, f'Page {document.page}')
            canvas.restoreState()
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        return stream.getvalue()
