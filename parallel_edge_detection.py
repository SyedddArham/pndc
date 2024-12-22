import numpy as np
from PIL import Image
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
from mpi4py import MPI
import psutil
import gc
from memory_profiler import profile
from multiprocessing import cpu_count, Pool
import argparse
from config import Config, load_config
from contextlib import contextmanager

@dataclass
class PerformanceMetrics:
    image_size: int
    num_processors: int
    total_time: float
    computation_time: float
    overhead_time: float
    speedup: float
    efficiency: float

@dataclass
class ExtendedPerformanceMetrics:
    image_size: int
    num_processors: int
    total_time: float
    computation_time: float
    overhead_time: float
    speedup: float
    efficiency: float
    memory_usage: float
    cpu_utilization: float
    communication_overhead: float
    quality_score: float

class EdgeDetectionOperators:
    @staticmethod
    def sobel():
        return {
            'x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        }
    
    @staticmethod
    def prewitt():
        return {
            'x': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            'y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        }
    
    @staticmethod
    def laplacian():
        return {
            'all': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        }

class ImageSizeExperiment:
    SIZES = [
        (512, 512),    # Small
        (1024, 1024),  # Medium
        (2048, 2048),  # Large
        (4096, 4096)   # Very Large
    ]
    
    def __init__(self):
        self.results = {}

    def preprocess_image_with_sizes(self, image_path):
        """Process same image with different sizes"""
        try:
            original = Image.open(image_path).convert('L')
            resized_images = {}
            
            for size in self.SIZES:
                resized = original.resize(size, Image.Resampling.LANCZOS)
                resized_images[size] = np.array(resized, dtype=np.float32)
            
            return resized_images
        except Exception as e:
            logging.error(f"Error preprocessing image sizes: {str(e)}")
            return None

class ValidationError(Exception):
    pass

class ProcessingError(Exception):
    pass

class DistributedEdgeDetector:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.operators = EdgeDetectionOperators()
        self.performance_metrics = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('DistributedEdgeDetector')

    def preprocess_image(self, image_path, target_size=(1024, 1024)):
        """
        Standardize image size and convert to grayscale
        """
        try:
            image = Image.open(image_path)
            image = image.convert('L')  # Convert to grayscale
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            return np.array(image, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None

    @profile
    def process_partition_with_memory_management(self, partition_data):
        """
        Process partition with memory monitoring and management
        """
        try:
            partition, start, end, overlap, operator_type = partition_data
            
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Process in chunks if partition is too large
            chunk_size = 1000
            if partition.shape[0] > chunk_size:
                results = []
                for i in range(0, partition.shape[0], chunk_size):
                    chunk = partition[i:i + chunk_size]
                    result = self.process_chunk(chunk, operator_type)
                    results.append(result)
                    gc.collect()  # Force garbage collection
                edges = np.vstack(results)
            else:
                edges = self.process_chunk(partition, operator_type)
            
            # Calculate memory usage
            final_memory = process.memory_info().rss
            memory_used = (final_memory - initial_memory) / 1024 / 1024  # MB
            
            return edges, memory_used
        except Exception as e:
            self.logger.error(f"Error processing partition: {str(e)}")
            return None

    def distribute_image(self, image):
        """
        Distribute image partitions using MPI
        """
        if self.rank == 0:
            partitions = self.create_overlapping_partitions(image, self.size)
        else:
            partitions = None
            
        # Scatter partitions to all processes
        partition_data = self.comm.scatter(partitions, root=0)
        return partition_data

    def gather_results(self, local_result):
        """
        Gather results from all processes
        """
        all_results = self.comm.gather(local_result, root=0)
        if self.rank == 0:
            return np.vstack([r[0] for r in all_results if r is not None])
        return None

    def validate_results(self, processed_image):
        """
        Validate edge detection quality
        """
        if processed_image is None:
            return 0.0
            
        # Basic quality metrics
        edge_density = np.mean(processed_image > 0)
        edge_continuity = np.std(processed_image[processed_image > 0])
        quality_score = (edge_density + 1/edge_continuity) / 2
        
        return quality_score

    def get_image_files(self, input_dir):
        """Get list of image files from input directory."""
        input_path = Path(input_dir)
        return list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

    def create_overlapping_partitions(self, image, num_processors, overlap=1):
        """Create partitions with overlapping borders."""
        height = image.shape[0]
        partition_height = height // num_processors
        partitions = []
        
        for i in range(num_processors):
            start = max(0, i * partition_height - overlap)
            end = min(height, (i + 1) * partition_height + overlap)
            partitions.append((image[start:end], start, end))
        
        return partitions

    def process_partition(self, args):
        """Process a single partition of the image with overlap handling."""
        partition, start, end, overlap, operator_type = args
        start_time = time.time()
        
        # Process partition using selected operator
        if operator_type == 'sobel':
            operators = self.operators.sobel()
            edge_x = self.apply_operator(partition, operators['x'])
            edge_y = self.apply_operator(partition, operators['y'])
            edges = np.sqrt(edge_x**2 + edge_y**2)
        elif operator_type == 'prewitt':
            operators = self.operators.prewitt()
            edge_x = self.apply_operator(partition, operators['x'])
            edge_y = self.apply_operator(partition, operators['y'])
            edges = np.sqrt(edge_x**2 + edge_y**2)
        elif operator_type == 'laplacian':
            operators = self.operators.laplacian()
            edges = np.abs(self.apply_operator(partition, operators['all']))
        
        # Remove overlap regions
        if start > 0:
            edges = edges[overlap:]
        if end < partition.shape[0]:
            edges = edges[:-overlap]
        
        computation_time = time.time() - start_time
        return edges, computation_time

    def apply_operator(self, partition, operator):
        """Apply convolution operator to partition."""
        height, width = partition.shape
        result = np.zeros((height-2, width-2))
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                window = partition[i-1:i+2, j-1:j+2]
                result[i-1, j-1] = np.sum(window * operator)
        
        return result

    def visualize_partition(self, image_shape, num_processors, save_path):
        """Visualize how the image is partitioned across processors."""
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(131)
        plt.imshow(Image.open(save_path).convert('L'), cmap='gray')
        plt.title('Original Image')
        
        # Plot partitioning visualization
        plt.subplot(132)
        partition_map = np.zeros(image_shape)
        height_per_partition = image_shape[0] // num_processors
        for i in range(num_processors):
            partition_map[i * height_per_partition:(i + 1) * height_per_partition] = i
        plt.imshow(partition_map, cmap='nipy_spectral')
        plt.colorbar(label='Processor ID')
        plt.title('Image Partitioning')
        
        # Plot final edge detection result
        plt.subplot(133)
        plt.imshow(Image.open(save_path.replace('.png', '_edges.png')), cmap='gray')
        plt.title('Edge Detection Result')
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_visualization.png'))
        plt.close()

    def measure_serial_performance(self, image_path, operator_type):
        """Measure performance of serial implementation."""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return 0.0
            
            start_time = time.time()
            
            # Process the entire image serially
            if operator_type == 'sobel':
                operators = self.operators.sobel()
                edge_x = self.apply_operator(image, operators['x'])
                edge_y = self.apply_operator(image, operators['y'])
                _ = np.sqrt(edge_x**2 + edge_y**2)
            elif operator_type == 'prewitt':
                operators = self.operators.prewitt()
                edge_x = self.apply_operator(image, operators['x'])
                edge_y = self.apply_operator(image, operators['y'])
                _ = np.sqrt(edge_x**2 + edge_y**2)
            elif operator_type == 'laplacian':
                operators = self.operators.laplacian()
                _ = np.abs(self.apply_operator(image, operators['all']))
            
            return time.time() - start_time
        except Exception as e:
            self.logger.error(f"Error in serial performance measurement: {str(e)}")
            return 0.0

    def determine_optimal_processors(self, image_path, max_processors=None):
        """
        Determine optimal number of processors based on image size and performance metrics
        Returns: optimal processor count
        """
        if max_processors is None:
            max_processors = min(cpu_count(), 4)  # Limit to 4 processors initially

        # Get image size
        if isinstance(image_path, (tuple, list)):
            # If image_path is already a size tuple
            image_size = image_path
        else:
            # If image_path is a path, load the image to get size
            try:
                with Image.open(image_path) as img:
                    image_size = img.size
            except Exception as e:
                self.logger.error(f"Error getting image size: {str(e)}")
                return 2  # Default to 2 processors on error

        # Calculate total pixels
        total_pixels = image_size[0] * image_size[1]
        
        # More conservative minimum pixels per processor
        MIN_PIXELS_PER_PROCESSOR = 262144  # 512x512
        
        # Calculate theoretical max processors based on image size
        theoretical_max = max(1, total_pixels // MIN_PIXELS_PER_PROCESSOR)
        
        # More conservative processor count
        recommended = min(
            theoretical_max,
            max_processors,
            2 if total_pixels < 1048576 else 4  # Use 2 processors for smaller images
        )
        
        self.logger.info(f"Recommended processor count for image size {image_size}: {recommended}")
        return recommended

    def process_image(self, image_path, output_path, operator_type='sobel', num_processors=None):
        """Process image with error handling, validation, and visualization"""
        try:
            # Get image size dynamically
            with Image.open(image_path) as img:
                original_size = img.size
                print(f"\nProcessing image of size: {original_size[0]}x{original_size[1]}")

            # Validate inputs
            self.validate_input(image_path)
            self.validate_operator(operator_type)
            
            # Process image with resource management
            with self.resource_context():
                metrics = self._process_image_impl(
                    image_path, output_path, operator_type, num_processors
                )
                
                if metrics:
                    # Print performance metrics
                    print(f"Optimal Processors: {metrics.num_processors}")
                    print(f"Total Time: {metrics.total_time:.2f}s")
                    print(f"Speedup: {metrics.speedup:.2f}x")
                    print(f"Efficiency: {metrics.efficiency:.2f}")
                    
                    if self.rank == 0:
                        # Generate visualizations for this processing run
                        self._generate_processing_visualization(
                            image_path,
                            output_path,
                            metrics,
                            original_size
                        )
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return None

    def _generate_processing_visualization(self, image_path, output_path, metrics, image_size):
        """Generate visualization for a single image processing run"""
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(131)
        plt.imshow(Image.open(image_path).convert('L'), cmap='gray')
        plt.title('Original Image')
        
        # Plot partitioning visualization
        plt.subplot(132)
        partition_map = np.zeros(image_size)
        height_per_partition = image_size[0] // metrics.num_processors
        for i in range(metrics.num_processors):
            partition_map[i * height_per_partition:(i + 1) * height_per_partition] = i
        plt.imshow(partition_map, cmap='nipy_spectral')
        plt.colorbar(label='Processor ID')
        plt.title('Image Partitioning')
        
        # Plot final edge detection result
        plt.subplot(133)
        plt.imshow(Image.open(output_path), cmap='gray')
        plt.title('Edge Detection Result')
        
        # Add performance metrics as text
        plt.figtext(0.02, 0.02, f'Speedup: {metrics.speedup:.2f}x\nEfficiency: {metrics.efficiency:.2f}', 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Save visualization
        viz_path = output_path.replace('.png', '_visualization.png')
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        
        # Generate performance bar chart
        plt.figure(figsize=(8, 6))
        metrics_data = {
            'Computation': metrics.computation_time,
            'Overhead': metrics.overhead_time
        }
        plt.bar(metrics_data.keys(), metrics_data.values())
        plt.title('Processing Time Breakdown')
        plt.ylabel('Time (seconds)')
        
        # Save performance chart
        perf_path = output_path.replace('.png', '_performance.png')
        plt.tight_layout()
        plt.savefig(perf_path)
        plt.close()

    def plot_performance_metrics(self, save_dir):
        """Generate performance analysis plots."""
        if self.performance_metrics:
            metrics_df = pd.DataFrame([vars(m) for m in self.performance_metrics])
            
            # Create performance plots directory
            plots_dir = Path(save_dir) / "performance_plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Speedup plot
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=metrics_df, x='num_processors', y='speedup', 
                        hue='image_size', marker='o')
            plt.title('Speedup vs Number of Processors')
            plt.xlabel('Number of Processors')
            plt.ylabel('Speedup')
            plt.savefig(plots_dir / 'speedup.png')
            plt.close()
            
            # Efficiency plot
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=metrics_df, x='num_processors', y='efficiency',
                        hue='image_size', marker='o')
            plt.title('Efficiency vs Number of Processors')
            plt.xlabel('Number of Processors')
            plt.ylabel('Efficiency')
            plt.savefig(plots_dir / 'efficiency.png')
            plt.close()
            
            # Time breakdown
            plt.figure(figsize=(12, 6))
            metrics_df_melted = metrics_df.melt(
                id_vars=['image_size', 'num_processors'],
                value_vars=['computation_time', 'overhead_time'],
                var_name='time_type', value_name='time'
            )
            sns.barplot(data=metrics_df_melted, x='num_processors', y='time',
                       hue='time_type')
            plt.title('Time Breakdown by Number of Processors')
            plt.xlabel('Number of Processors')
            plt.ylabel('Time (seconds)')
            plt.savefig(plots_dir / 'time_breakdown.png')
            plt.close()
            
            # Save metrics to JSON
            with open(plots_dir / 'performance_metrics.json', 'w') as f:
                json.dump([vars(m) for m in self.performance_metrics], f, indent=2)

    def monitor_performance(self, start_time, end_time, image_size, memory_used):
        """
        Monitor and record detailed performance metrics
        """
        total_time = end_time - start_time
        
        # Get CPU utilization
        cpu_percent = psutil.cpu_percent()
        
        # Calculate communication overhead (MPI specific)
        comm_start = time.time()
        self.comm.Barrier()
        communication_overhead = time.time() - comm_start
        
        metrics = ExtendedPerformanceMetrics(
            image_size=image_size,
            num_processors=self.size,
            total_time=total_time,
            computation_time=total_time - communication_overhead,
            overhead_time=communication_overhead,
            speedup=self.measure_serial_performance() / total_time,
            efficiency=(self.measure_serial_performance() / total_time) / self.size,
            memory_usage=memory_used,
            cpu_utilization=cpu_percent,
            communication_overhead=communication_overhead,
            quality_score=self.validate_results(self.current_result)
        )
        
        self.performance_metrics.append(metrics)
        return metrics

    def generate_performance_report(self, save_dir):
        """
        Generate comprehensive performance report
        """
        if not self.performance_metrics:
            return
            
        report_dir = Path(save_dir) / "performance_report"
        report_dir.mkdir(exist_ok=True)
        
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame([vars(m) for m in self.performance_metrics])
        
        # Generate plots
        self._plot_performance_metrics(metrics_df, report_dir)
        
        # Save detailed report
        self._save_detailed_report(metrics_df, report_dir)

    def run_size_experiments(self, image_path, operator_type='sobel'):
        """Run experiments with different image sizes"""
        experiment = ImageSizeExperiment()
        resized_images = experiment.preprocess_image_with_sizes(image_path)
        
        if not resized_images:
            return None
            
        size_results = {}
        for size, image in resized_images.items():
            # Determine optimal processor count for this size
            num_processors = self.determine_optimal_processors(size)
            
            # Process image
            start_time = time.time()
            result = self.process_image_data(
                image, 
                operator_type, 
                num_processors
            )
            
            # Record metrics
            metrics = self.calculate_metrics(
                size, 
                num_processors,
                time.time() - start_time,
                result
            )
            
            size_results[size] = metrics
            
        return size_results

    def generate_detailed_report(self, save_dir, experiment_results):
        """Generate comprehensive performance report"""
        report_dir = Path(save_dir) / "detailed_report"
        report_dir.mkdir(exist_ok=True)
        
        # Create report sections
        sections = {
            'summary': self._generate_summary(),
            'size_analysis': self._analyze_image_sizes(experiment_results),
            'processor_analysis': self._analyze_processor_scaling(),
            'memory_analysis': self._analyze_memory_usage(),
            'recommendations': self._generate_recommendations()
        }
        
        # Generate plots
        self._generate_report_plots(report_dir, experiment_results)
        
        # Save detailed report as HTML
        self._save_html_report(report_dir, sections)
        
        # Save raw data as JSON
        with open(report_dir / 'raw_data.json', 'w') as f:
            json.dump(experiment_results, f, indent=2)

    def _generate_report_plots(self, report_dir, results):
        """Generate comprehensive performance plots"""
        # Size vs Performance Plot
        plt.figure(figsize=(12, 6))
        sizes = sorted(results.keys())
        speedups = [results[size].speedup for size in sizes]
        efficiencies = [results[size].efficiency for size in sizes]
        
        plt.subplot(121)
        plt.plot(sizes, speedups, 'o-', label='Speedup')
        plt.xlabel('Image Size')
        plt.ylabel('Speedup')
        plt.title('Image Size vs Speedup')
        
        plt.subplot(122)
        plt.plot(sizes, efficiencies, 'o-', label='Efficiency')
        plt.xlabel('Image Size')
        plt.ylabel('Efficiency')
        plt.title('Image Size vs Efficiency')
        
        plt.tight_layout()
        plt.savefig(report_dir / 'size_performance.png')
        plt.close()

    def _generate_summary(self):
        """Generate performance summary"""
        return {
            'total_images_processed': len(self.performance_metrics),
            'average_speedup': np.mean([m.speedup for m in self.performance_metrics]),
            'average_efficiency': np.mean([m.efficiency for m in self.performance_metrics]),
            'best_performance': max(self.performance_metrics, key=lambda x: x.speedup),
            'memory_usage': {
                'average': np.mean([m.memory_usage for m in self.performance_metrics]),
                'peak': max([m.memory_usage for m in self.performance_metrics])
            }
        }

    def _save_html_report(self, report_dir, sections):
        """Save report as HTML with interactive plots"""
        html_content = f"""
        <html>
        <head>
            <title>Edge Detection Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .plot {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Edge Detection Performance Analysis</h1>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <p>Total Images Processed: {sections['summary']['total_images_processed']}</p>
                <p>Average Speedup: {sections['summary']['average_speedup']:.2f}x</p>
                <p>Average Efficiency: {sections['summary']['average_efficiency']:.2f}</p>
            </div>
            
            <div class="section">
                <h2>Image Size Analysis</h2>
                <img src="size_performance.png" class="plot">
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {sections['recommendations']}
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_dir / 'report.html', 'w') as f:
            f.write(html_content)

    def validate_input(self, image_path: str) -> None:
        """Validate input image"""
        if not Path(image_path).exists():
            raise ValidationError(f"Image file not found: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                if img.size[0] < 32 or img.size[1] < 32:
                    raise ValidationError(f"Image too small: {image_path}")
        except Exception as e:
            raise ValidationError(f"Invalid image file: {str(e)}")

    def validate_operator(self, operator_type: str) -> None:
        """Validate operator type"""
        valid_operators = ['sobel', 'prewitt', 'laplacian']
        if operator_type not in valid_operators:
            raise ValidationError(f"Invalid operator type: {operator_type}")

    @contextmanager
    def resource_context(self):
        """Context manager for resource cleanup"""
        try:
            yield
        finally:
            gc.collect()
            if hasattr(self, 'pool'):
                self.pool.close()
                self.pool.join()

    def _process_image_impl(self, image_path, output_path, operator_type='sobel', num_processors=None):
        """
        Implementation of image processing logic
        """
        if num_processors is None:
            num_processors = self.determine_optimal_processors(image_path)
        
        # Load and preprocess image
        image = self.preprocess_image(image_path)
        if image is None:
            raise ProcessingError(f"Failed to preprocess image: {image_path}")
            
        original_shape = image.shape
        
        # Create overlapping partitions
        overlap = 1
        partitions = self.create_overlapping_partitions(image, num_processors, overlap)
        args = [(partition, start, end, overlap, operator_type) 
                for partition, start, end in partitions]
        
        # Process partitions in parallel
        start_time = time.time()
        with Pool(num_processors) as pool:
            results = pool.map(self.process_partition, args)
        
        # Separate results and timing
        processed_parts, computation_times = zip(*results)
        
        # Combine results
        final_image = np.vstack(processed_parts)
        
        # Calculate timing metrics
        total_time = time.time() - start_time
        computation_time = max(computation_times)
        overhead_time = total_time - computation_time
        
        # Measure serial performance for comparison
        serial_time = self.measure_serial_performance(image_path, operator_type)
        
        # Calculate performance metrics
        speedup = serial_time / total_time if total_time > 0 else 0
        efficiency = speedup / num_processors if num_processors > 0 else 0
        
        # Save result
        output_image = Image.fromarray(final_image * 255 / np.max(final_image))
        output_image = output_image.convert('L')
        output_image.save(output_path)
        
        return PerformanceMetrics(
            image_size=original_shape[0],
            num_processors=num_processors,
            total_time=total_time,
            computation_time=computation_time,
            overhead_time=overhead_time,
            speedup=speedup,
            efficiency=efficiency
        )

def setup_logging(log_level: str):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('edge_detection.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Edge Detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--input-dir', type=str,
                      help='Input directory containing images')
    parser.add_argument('--output-dir', type=str,
                      help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.log_level)
    
    config = load_config(args.config)
    
    detector = DistributedEdgeDetector()
    
    if detector.rank == 0:
        image_files = detector.get_image_files(config.input_dir)
        results_dir = Path(config.output_dir)
        results_dir.mkdir(exist_ok=True)
        
        operators = config.operators
        # Let the algorithm determine optimal processor counts
        total_operations = len(image_files) * len(operators)
        operation_count = 0
        
        all_metrics = []  # Store metrics for final visualization
        
        for image_path in image_files:
            image_name = Path(image_path).stem
            
            for operator in operators:
                operation_count += 1
                print(f"\nProgress: {operation_count}/{total_operations}")
                
                # Let the algorithm determine optimal processor count
                output_path = results_dir / f"{image_name}_{operator}_auto.png"
                
                metrics = detector.process_image(
                    image_path, 
                    str(output_path), 
                    operator
                )
                
                if metrics:
                    all_metrics.append(metrics)
        
        # Generate final summary visualization
        if all_metrics:
            detector.plot_performance_metrics(results_dir, all_metrics)
            
        print("\nProcessing complete! Check the 'results' directory for visualizations.")

if __name__ == "__main__":
    main()