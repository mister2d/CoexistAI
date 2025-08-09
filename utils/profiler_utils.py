# Profiling system for query_web_response
import time
import datetime
import logging
from collections import defaultdict

# Set up logger
logger = logging.getLogger(__name__)
class WebSearchProfiler:
    """
    Comprehensive profiling system for web search operations.
    Tracks timing, memory usage, and performance metrics for each step.
    """
    
    def __init__(self, query):
        self.query = query
        self.start_time = time.time()
        self.checkpoints = []
        self.step_times = defaultdict(float)
        self.step_counts = defaultdict(int)
        self.current_step = None
        self.step_start_time = None
        self.total_urls_processed = 0
        self.successful_urls = 0
        self.failed_urls = 0
        self.total_docs_retrieved = 0
        self.context_length = 0
        
        # Time saved tracking
        self.websites_processed = 0
        self.total_words_processed = 0
        self.url_word_counts = {}  # Track words per URL
        self.HUMAN_READING_SPEED_WPM = 250  # Words per minute
        
        # URL-level processing tracking
        self.url_processing_times = {}  # {url: processing_time}
        self.url_details = {}  # {url: {docs_count, context_length, status, error}}
        self.url_start_times = {}  # Track when URL processing started
        
    def start_step(self, step_name, details=""):
        """Start timing a specific step"""
        if self.current_step:
            self.end_step()
        
        self.current_step = step_name
        self.step_start_time = time.time()
        self.step_counts[step_name] += 1
        
        checkpoint = {
            'step': step_name,
            'start_time': self.step_start_time,
            'details': details,
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        }
        self.checkpoints.append(checkpoint)
        logger.info(f"🔍 PROFILER [{self.query[:50]}...]: Starting {step_name} - {details}")
        
    def end_step(self, additional_info=""):
        """End timing the current step"""
        if not self.current_step or not self.step_start_time:
            return
            
        step_duration = time.time() - self.step_start_time
        self.step_times[self.current_step] += step_duration
        
        # Update the last checkpoint with end time and duration
        if self.checkpoints:
            self.checkpoints[-1].update({
                'end_time': time.time(),
                'duration': step_duration,
                'additional_info': additional_info
            })
        
        logger.info(f"⏱️  PROFILER [{self.query[:50]}...]: Completed {self.current_step} in {step_duration:.3f}s - {additional_info}")
        
        self.current_step = None
        self.step_start_time = None
        
    def add_metric(self, metric_name, value):
        """Add a custom metric"""
        if metric_name == 'urls_processed':
            self.total_urls_processed += value
        elif metric_name == 'successful_urls':
            self.successful_urls += value
        elif metric_name == 'failed_urls':
            self.failed_urls += value
        elif metric_name == 'docs_retrieved':
            self.total_docs_retrieved += value
        elif metric_name == 'context_length':
            self.context_length = value
    
    def add_url_content(self, url, content):
        """Add content from a processed URL for time saved calculation"""
        if content and url:
            # Count words in the content (simple word count by splitting on whitespace)
            word_count = len(content.split())
            self.url_word_counts[url] = word_count
            self.total_words_processed += word_count
            self.websites_processed += 1
            logger.debug(f"Added {word_count} words from {url} (Total: {self.total_words_processed} words)")
    
    def start_url_processing(self, url):
        """Start tracking processing time for a specific URL"""
        self.url_start_times[url] = time.time()
        logger.debug(f"Started tracking URL processing: {url}")
    
    def end_url_processing(self, url, docs_count=0, context_length=0, status="success", error=None):
        """End tracking processing time for a specific URL"""
        if url in self.url_start_times:
            processing_time = time.time() - self.url_start_times[url]
            self.url_processing_times[url] = processing_time
            self.url_details[url] = {
                'docs_count': docs_count,
                'context_length': context_length,
                'status': status,
                'error': error,
                'processing_time': processing_time
            }
            del self.url_start_times[url]  # Clean up
            logger.debug(f"Completed URL processing: {url} in {processing_time:.3f}s")
        else:
            logger.warning(f"URL processing end called without start: {url}")
    
    def calculate_time_saved(self):
        """Calculate time saved by CoexistAI vs manual reading"""
        if self.total_words_processed == 0:
            return 0, 0, 0  # minutes, hours, days
        
        # Calculate time to read all content at human reading speed
        reading_time_minutes = self.total_words_processed / self.HUMAN_READING_SPEED_WPM
        reading_time_hours = reading_time_minutes / 60
        reading_time_days = reading_time_hours / 24
        
        return reading_time_minutes, reading_time_hours, reading_time_days
    
    def get_time_saved_summary(self):
        """Get a formatted summary of time saved"""
        minutes, hours, days = self.calculate_time_saved()
        
        if minutes < 1:
            return "Less than 1 minute of reading time saved"
        elif minutes < 60:
            return f"{minutes:.1f} minutes of reading time saved"
        elif hours < 24:
            return f"{hours:.1f} hours ({minutes:.0f} minutes) of reading time saved"
        else:
            return f"{days:.1f} days ({hours:.1f} hours) of reading time saved"
            
    def get_summary(self):
        """Generate a comprehensive profiling summary"""
        if self.current_step:
            self.end_step("Final step")
            
        total_time = time.time() - self.start_time
        
        # Calculate time saved
        time_saved_summary = self.get_time_saved_summary()
        minutes, hours, days = self.calculate_time_saved()
        
        summary = [
            "\n" + "="*80,
            f"🔍 WEB SEARCH PROFILING REPORT - Query: {self.query[:60]}...",
            "="*80,
            f"⏰ Total Execution Time: {total_time:.3f} seconds",
            f"📊 Total Steps Executed: {len(self.checkpoints)}",
            f"🌐 URLs Processed: {self.total_urls_processed} (✅ {self.successful_urls}, ❌ {self.failed_urls})",
            f"📄 Documents Retrieved: {self.total_docs_retrieved}",
            f"📝 Final Context Length: {self.context_length} characters",
            "",
            "⏱️  TIME SAVED ANALYSIS:",
            "-" * 50,
            f"📚 Websites Read for You: {self.websites_processed}",
            f"📖 Total Words Processed: {self.total_words_processed:,}",
            f"🚀 Time Saved: {time_saved_summary}",
            f"⚡ Efficiency: {total_time/60:.1f} min processing vs {minutes:.1f} min manual reading",
            f"📈 Speed Multiplier: {minutes/max(total_time/60, 0.01):.1f}x faster than manual reading",
            "",
            "🚀 7-STEP PIPELINE BREAKDOWN:",
            "-" * 50
        ]
        
        # Add detailed 7-step report
        summary.extend(self._get_seven_step_report(total_time))
        
        summary.extend([
            "",
            "🌐 URL-LEVEL PROCESSING REPORT:",
            "-" * 50
        ])
        
        # Add URL-level processing report
        summary.extend(self._get_url_processing_report())
        
        summary.extend([
            "",
            "📈 COMPLETE STEP-BY-STEP BREAKDOWN:",
            "-" * 50
        ])
        
        # Sort steps by total time spent
        sorted_steps = sorted(self.step_times.items(), key=lambda x: x[1], reverse=True)
        
        for step_name, total_time_spent in sorted_steps:
            count = self.step_counts[step_name]
            avg_time = total_time_spent / count if count > 0 else 0
            percentage = (total_time_spent / total_time) * 100 if total_time > 0 else 0
            
            summary.append(f"  {step_name:.<30} {total_time_spent:>8.3f}s ({percentage:>5.1f}%) [{count}x, avg: {avg_time:.3f}s]")
        
        summary.extend([
            "",
            "🕐 DETAILED TIMELINE:",
            "-" * 50
        ])
        
        for i, checkpoint in enumerate(self.checkpoints, 1):
            duration = checkpoint.get('duration', 0)
            details = checkpoint.get('details', '')
            additional_info = checkpoint.get('additional_info', '')
            timestamp = checkpoint.get('timestamp', '')
            
            info_str = f" - {details}" if details else ""
            additional_str = f" | {additional_info}" if additional_info else ""
            
            summary.append(f"  {i:2d}. [{timestamp}] {checkpoint['step']:.<25} {duration:>8.3f}s{info_str}{additional_str}")
        
        summary.extend([
            "",
            "🎯 PERFORMANCE INSIGHTS:",
            "-" * 50
        ])
        
        # Add performance insights
        if self.total_urls_processed > 0:
            success_rate = (self.successful_urls / self.total_urls_processed) * 100
            summary.append(f"  • URL Success Rate: {success_rate:.1f}%")
            
        if 'search_execution' in self.step_times and 'context_generation' in self.step_times:
            search_time = self.step_times['search_execution']
            context_time = self.step_times['context_generation']
            if search_time > 0 and context_time > 0:
                ratio = context_time / search_time
                summary.append(f"  • Context/Search Time Ratio: {ratio:.2f}x")
        
        if self.context_length > 0 and total_time > 0:
            chars_per_second = self.context_length / total_time
            summary.append(f"  • Processing Speed: {chars_per_second:.0f} chars/second")
            
        summary.append("="*80 + "\n")
        
        return "\n".join(summary)
    
    def _get_seven_step_report(self, total_time):
        """Generate detailed 7-step pipeline report""" 
        # Define the 7 main pipeline steps in order
        main_steps = [
            ("1️⃣", "query_agent", "Query Analysis", "Generating search queries from user input"),
            ("2️⃣", "web_search_execution", "Web Search", "Executing web search and URL extraction"),
            ("3️⃣", "url_collection", "URL Collection", "Collecting and preparing URLs for processing"),
            ("4️⃣", "parallel_url_processing", "URL Processing", "Processing URLs in parallel"),
            ("5️⃣", "context_building", "Context Building", "Building final context from processed documents"),
            ("6️⃣", "context_generation", "Context Generation", "Processing URLs and generating context"),
            ("7️⃣", "response_generation", "Response Generation", "Generating final response from context")
        ]
        
        report = []
        total_main_time = 0
        
        # Calculate total time for main steps
        for _, step_key, _, _ in main_steps:
            if step_key in self.step_times:
                total_main_time += self.step_times[step_key]
        
        # Generate report for each step
        for emoji, step_key, step_name, description in main_steps:
            step_time = self.step_times.get(step_key, 0)
            step_count = self.step_counts.get(step_key, 0)
            
            if step_count > 0:
                percentage = (step_time / total_time) * 100 if total_time > 0 else 0
                avg_time = step_time / step_count
                
                # Create visual progress bar
                bar_length = 20
                filled_length = int(bar_length * percentage / 100) if percentage <= 100 else bar_length
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                # Status indicator
                if step_time > 0:
                    status = "✅"
                    time_str = f"{step_time:.3f}s"
                else:
                    status = "⏸️"
                    time_str = "0.000s"
                
                report.extend([
                    f"{emoji} {step_name:.<20} {status} {time_str:>8} ({percentage:>5.1f}%) [{step_count}x]",
                    f"   📊 [{bar}] {description}",
                    f"   ⚡ Avg: {avg_time:.3f}s/execution",
                    ""
                ])
            else:
                # Step was not executed
                bar = "░" * 20
                report.extend([
                    f"{emoji} {step_name:.<20} ⏸️  0.000s ( 0.0%) [0x]",
                    f"   📊 [{bar}] {description}",
                    f"   ⚡ Not executed",
                    ""
                ])
        
        # Add pipeline summary
        pipeline_efficiency = (total_main_time / total_time) * 100 if total_time > 0 else 0
        other_time = total_time - total_main_time
        
        report.extend([
            f"📋 PIPELINE SUMMARY:",
            f"   • Main Steps Time: {total_main_time:.3f}s ({pipeline_efficiency:.1f}% of total)",
            f"   • Other Operations: {other_time:.3f}s ({100-pipeline_efficiency:.1f}% of total)",
            f"   • Pipeline Efficiency: {pipeline_efficiency:.1f}%"
        ])
        
        return report
    
    def _get_url_processing_report(self):
        """Generate detailed URL-level processing report"""
        if not self.url_processing_times:
            return ["   📝 No URL processing data available"]
        
        report = []
        
        # Sort URLs by processing time (slowest first)
        sorted_urls = sorted(
            self.url_processing_times.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        total_url_time = sum(self.url_processing_times.values())
        
        # Add summary stats
        report.extend([
            f"📊 URL PROCESSING SUMMARY:",
            f"   • Total URLs Processed: {len(self.url_processing_times)}",
            f"   • Total URL Processing Time: {total_url_time:.3f}s",
            f"   • Average Time per URL: {total_url_time/len(self.url_processing_times):.3f}s",
            f"   • Fastest URL: {min(self.url_processing_times.values()):.3f}s",
            f"   • Slowest URL: {max(self.url_processing_times.values()):.3f}s",
            ""
        ])
        
        # Add individual URL details
        report.append("🔗 INDIVIDUAL URL PERFORMANCE:")
        
        for i, (url, processing_time) in enumerate(sorted_urls, 1):
            details = self.url_details.get(url, {})
            docs_count = details.get('docs_count', 0)
            context_length = details.get('context_length', 0)
            status = details.get('status', 'unknown')
            error = details.get('error')
            
            # Determine status emoji and color
            if status == 'success':
                status_emoji = "✅"
            elif status == 'failed':
                status_emoji = "❌"
            elif status == 'timeout':
                status_emoji = "⏰"
            else:
                status_emoji = "❓"
            
            # Create performance indicator
            if processing_time < 2.0:
                speed_emoji = "🚀"  # Fast
            elif processing_time < 5.0:
                speed_emoji = "🔄"  # Medium
            else:
                speed_emoji = "🐌"  # Slow
            
            # Truncate URL for display
            display_url = url if len(url) <= 60 else url[:57] + "..."
            
            report.append(
                f"   {i:2d}. {status_emoji} {speed_emoji} {processing_time:>6.3f}s | "
                f"Docs: {docs_count:2d} | Context: {context_length:4d} chars"
            )
            report.append(f"       🔗 {display_url}")
            
            if error:
                report.append(f"       ⚠️  Error: {str(error)[:80]}..." if len(str(error)) > 80 else f"       ⚠️  Error: {error}")
            
            report.append("")  # Empty line for spacing
        
        # Add performance insights
        if len(sorted_urls) > 1:
            slowest_time = sorted_urls[0][1]
            fastest_time = sorted_urls[-1][1]
            speed_ratio = slowest_time / fastest_time if fastest_time > 0 else float('inf')
            
            report.extend([
                "📈 URL PERFORMANCE INSIGHTS:",
                f"   • Speed Variation: {speed_ratio:.1f}x difference between fastest and slowest",
                f"   • Slowest URL took {slowest_time:.3f}s ({(slowest_time/total_url_time)*100:.1f}% of total URL time)",
                f"   • Top 3 URLs account for {sum(time for _, time in sorted_urls[:3])/total_url_time*100:.1f}% of processing time"
            ])
            
            # Identify potential bottlenecks
            slow_urls = [url for url, time in sorted_urls if time > total_url_time/len(sorted_urls) * 2]
            if slow_urls:
                report.append(f"   ⚠️  {len(slow_urls)} URLs are significantly slower than average")
        
        return report
    
    def print_summary(self):
        """Print the profiling summary"""
        print(self.get_summary())
        
# Global profiler instance
_current_profiler = None

def get_profiler():
    """Get the current profiler instance"""
    return _current_profiler

def set_profiler(profiler):
    """Set the current profiler instance"""
    global _current_profiler
    _current_profiler = profiler

