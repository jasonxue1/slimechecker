use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;

const R_CHUNK: i32 = 8;
const MASK_WIDTH: usize = R_CHUNK as usize * 2 + 1;
const BLOCKS_PER_CHUNK: i32 = 16;
const SCAN_BLOCK_SIZE: i32 = 64;

const BASE_MASK_CONFIG: MaskConfig = MaskConfig {
    world_seed: 7_584_197_480_721_263_469,
    despawn_sphere: true,
    exclusion_sphere: true,
    y_offset: 0,
    chunk_weight: 5,
};

#[derive(Debug, Parser)]
#[command(
    name = "slimechecker",
    about = "Fine-search validator for candidate slime chunk centers"
)]
struct Cli {
    /// Input CSV that lists chunkX,chunkZ[,chunkCount]
    #[arg(short, long)]
    input: PathBuf,
    /// Output CSV path (semicolon-delimited, same style as slimefinder)
    #[arg(short, long)]
    output: PathBuf,
    /// Minecraft world seed used when determining slime chunks
    #[arg(long = "world-seed", default_value_t = BASE_MASK_CONFIG.world_seed)]
    world_seed: i64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mask_cfg = MaskConfig {
        world_seed: cli.world_seed,
        ..BASE_MASK_CONFIG
    };
    let requests = read_requests(&cli.input)?;
    if requests.is_empty() {
        println!(
            "Input {} contains no chunk coordinates. Nothing to do.",
            cli.input.display()
        );
        return Ok(());
    }

    let in_points = build_fine_points();
    let template_cache = build_template_cache(&in_points, &mask_cfg);

    let mut results = Vec::new();
    let mut total_windows = 0usize;
    for request in &requests {
        let (count, best) = process_request(request, &mask_cfg, &template_cache)?;
        total_windows += count;
        results.push(SortedEntry {
            source_chunk: request.source_chunk_string(),
            chunk_count: request.chunk_count,
            mask: best,
        });
    }
    results.sort_by(|a, b| b.mask.block_size.cmp(&a.mask.block_size));

    let mut writer = csv::WriterBuilder::new()
        .delimiter(b';')
        .from_path(&cli.output)
        .with_context(|| format!("failed to create {}", cli.output.display()))?;
    writer.write_record([
        "source-chunk",
        "source-chunkCount",
        "block-position",
        "chunk-position",
        "blockSize",
        "chunkSize",
    ])?;
    for entry in &results {
        let mask = &entry.mask;
        writer.write_record([
            entry.source_chunk.clone(),
            format_chunk_count(entry.chunk_count),
            mask.block_string(),
            mask.chunk_string(),
            mask.block_ratio(),
            mask.chunk_ratio(),
        ])?;
    }
    writer.flush()?;
    println!(
        "Processed {} chunks ({} block positions) into {}",
        requests.len(),
        total_windows,
        cli.output.display()
    );
    Ok(())
}

fn read_requests(path: &Path) -> Result<Vec<ChunkRequest>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .trim(csv::Trim::All)
        .from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let mut requests = Vec::new();
    for (idx, record) in rdr.records().enumerate() {
        let rec = record?;
        if rec.len() == 0 || rec.iter().all(|field| field.trim().is_empty()) {
            continue;
        }
        if rec.len() < 2 {
            bail!(
                "line {} in {} does not contain chunkX,chunkZ",
                idx + 1,
                path.display()
            );
        }
        let chunk_x: i32 = rec[0].parse().with_context(|| {
            format!("invalid chunk X on line {} of {}", idx + 1, path.display())
        })?;
        let chunk_z: i32 = rec[1].parse().with_context(|| {
            format!("invalid chunk Z on line {} of {}", idx + 1, path.display())
        })?;
        let chunk_count = rec
            .get(2)
            .and_then(|s| if s.trim().is_empty() { None } else { Some(s) })
            .map(|s| {
                s.parse::<i32>().with_context(|| {
                    format!(
                        "invalid chunk count on line {} of {}",
                        idx + 1,
                        path.display()
                    )
                })
            })
            .transpose()?;
        requests.push(ChunkRequest {
            chunk_x,
            chunk_z,
            chunk_count,
        });
    }
    Ok(requests)
}

fn process_request(
    request: &ChunkRequest,
    mask_cfg: &MaskConfig,
    templates: &HashMap<Point, MaskTemplate>,
) -> Result<(usize, MaskData)> {
    let half = SCAN_BLOCK_SIZE / 2;
    let center_block_x = request.chunk_x * BLOCKS_PER_CHUNK + BLOCKS_PER_CHUNK / 2;
    let center_block_z = request.chunk_z * BLOCKS_PER_CHUNK + BLOCKS_PER_CHUNK / 2;
    let start_x = center_block_x - half;
    let start_z = center_block_z - half;
    let mut best: Option<MaskData> = None;
    let mut count = 0usize;

    for dx in 0..SCAN_BLOCK_SIZE {
        for dz in 0..SCAN_BLOCK_SIZE {
            let block_x = start_x + dx;
            let block_z = start_z + dz;
            let pos = Position::from_block(block_x, block_z);
            let template = templates
                .get(&pos.in_block)
                .expect("missing template for in-block coordinate");
            let data = compute_mask(
                mask_cfg,
                template,
                &MaskJob {
                    chunk: pos.chunk,
                    in_block: pos.in_block,
                },
            );
            match &best {
                Some(current) if current.block_size >= data.block_size => {}
                _ => best = Some(data),
            }
            count += 1;
        }
    }
    let best = best.ok_or_else(|| anyhow!("no results produced for chunk"))?;
    Ok((count, best))
}

#[derive(Debug, Clone, Copy)]
struct MaskConfig {
    world_seed: i64,
    despawn_sphere: bool,
    exclusion_sphere: bool,
    y_offset: i32,
    chunk_weight: u32,
}

#[derive(Debug)]
struct ChunkRequest {
    chunk_x: i32,
    chunk_z: i32,
    chunk_count: Option<i32>,
}

impl ChunkRequest {
    fn source_chunk_string(&self) -> String {
        format!("{},{}", self.chunk_x, self.chunk_z)
    }
}

struct SortedEntry {
    source_chunk: String,
    chunk_count: Option<i32>,
    mask: MaskData,
}

fn format_chunk_count(value: Option<i32>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(String::new)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Point {
    x: i32,
    z: i32,
}

impl Point {
    fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }
}

#[derive(Clone, Copy, Debug)]
struct Position {
    chunk: Point,
    in_block: Point,
}

impl Position {
    fn from_block(block_x: i32, block_z: i32) -> Self {
        let chunk_x = block_x.div_euclid(BLOCKS_PER_CHUNK);
        let chunk_z = block_z.div_euclid(BLOCKS_PER_CHUNK);
        let in_x = block_x.rem_euclid(BLOCKS_PER_CHUNK);
        let in_z = block_z.rem_euclid(BLOCKS_PER_CHUNK);
        Self {
            chunk: Point::new(chunk_x, chunk_z),
            in_block: Point::new(in_x, in_z),
        }
    }
}

#[derive(Clone, Debug)]
struct MaskTemplate {
    chunk_weights: [[u32; MASK_WIDTH]; MASK_WIDTH],
    block_surface_area: u32,
    chunk_surface_area: u32,
}

#[derive(Clone, Copy, Debug)]
struct MaskGeometry {
    r_exclusion_sq: i64,
    r_despawn_sq: i64,
}

impl MaskGeometry {
    fn from_config(cfg: &MaskConfig) -> Self {
        let y = cfg.y_offset as i64;
        let y_sq = y * y;
        let exclusion_limit = 24_i64 * 24_i64;
        let r_exclusion_sq = exclusion_limit - y_sq.min(exclusion_limit);
        let r_despawn_sq = 128_i64 * 128_i64 - y_sq;
        Self {
            r_exclusion_sq,
            r_despawn_sq,
        }
    }
}

fn build_fine_points() -> Vec<Point> {
    let mut v = Vec::with_capacity(256);
    for x in 0..BLOCKS_PER_CHUNK {
        for z in 0..BLOCKS_PER_CHUNK {
            v.push(Point::new(x, z));
        }
    }
    v
}

fn build_template_cache(points: &[Point], cfg: &MaskConfig) -> HashMap<Point, MaskTemplate> {
    let geometry = MaskGeometry::from_config(cfg);
    let mut map = HashMap::new();
    for point in points {
        map.entry(*point)
            .or_insert_with(|| build_template(cfg, &geometry, *point));
    }
    map
}

fn build_template(cfg: &MaskConfig, geometry: &MaskGeometry, in_block: Point) -> MaskTemplate {
    let mut chunk_weights = [[0u32; MASK_WIDTH]; MASK_WIDTH];
    let mut block_surface_area = 0u32;
    let mut chunk_surface_area = 0u32;
    for (dx_idx, chunk_x) in (-R_CHUNK..=R_CHUNK).enumerate() {
        for (dz_idx, chunk_z) in (-R_CHUNK..=R_CHUNK).enumerate() {
            let mut weight = 0u32;
            for local_x in 0..BLOCKS_PER_CHUNK {
                for local_z in 0..BLOCKS_PER_CHUNK {
                    let block_x = chunk_x * BLOCKS_PER_CHUNK + local_x;
                    let block_z = chunk_z * BLOCKS_PER_CHUNK + local_z;
                    if is_block_inside(cfg, geometry, in_block, block_x, block_z) {
                        weight += 1;
                    }
                }
            }
            chunk_weights[dx_idx][dz_idx] = weight;
            block_surface_area += weight;
            if weight > cfg.chunk_weight {
                chunk_surface_area += 1;
            }
        }
    }
    MaskTemplate {
        chunk_weights,
        block_surface_area,
        chunk_surface_area,
    }
}

fn is_block_inside(
    cfg: &MaskConfig,
    geometry: &MaskGeometry,
    in_block: Point,
    block_x: i32,
    block_z: i32,
) -> bool {
    let dx = block_x - in_block.x;
    let dz = block_z - in_block.z;
    let dsqr = (dx as i64) * (dx as i64) + (dz as i64) * (dz as i64);
    if cfg.despawn_sphere && dsqr > geometry.r_despawn_sq {
        return false;
    }
    if cfg.exclusion_sphere && dsqr <= geometry.r_exclusion_sq {
        return false;
    }
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct MaskJob {
    chunk: Point,
    in_block: Point,
}

fn compute_mask(cfg: &MaskConfig, template: &MaskTemplate, job: &MaskJob) -> MaskData {
    let mut block_size = 0u32;
    let mut chunk_size = 0u32;
    for (dx_idx, row) in template.chunk_weights.iter().enumerate() {
        let rel_x = dx_idx as i32 - R_CHUNK;
        for (dz_idx, &weight) in row.iter().enumerate() {
            if weight == 0 {
                continue;
            }
            let rel_z = dz_idx as i32 - R_CHUNK;
            if is_slime_chunk(cfg.world_seed, job.chunk.x + rel_x, job.chunk.z + rel_z) {
                block_size += weight;
                if weight > cfg.chunk_weight {
                    chunk_size += 1;
                }
            }
        }
    }
    MaskData {
        chunk: job.chunk,
        in_block: job.in_block,
        block_surface_area: template.block_surface_area,
        chunk_surface_area: template.chunk_surface_area,
        block_size,
        chunk_size,
    }
}

#[derive(Clone, Debug)]
struct MaskData {
    chunk: Point,
    in_block: Point,
    block_surface_area: u32,
    chunk_surface_area: u32,
    block_size: u32,
    chunk_size: u32,
}

impl MaskData {
    fn chunk_string(&self) -> String {
        format!(
            "{}c{},{}c{}",
            self.chunk.x, self.in_block.x, self.chunk.z, self.in_block.z
        )
    }

    fn block_string(&self) -> String {
        let block_x = self.chunk.x * BLOCKS_PER_CHUNK + self.in_block.x;
        let block_z = self.chunk.z * BLOCKS_PER_CHUNK + self.in_block.z;
        format!("{},{}", block_x, block_z)
    }

    fn block_ratio(&self) -> String {
        format!("{}/{}", self.block_size, self.block_surface_area)
    }

    fn chunk_ratio(&self) -> String {
        format!("{}/{}", self.chunk_size, self.chunk_surface_area)
    }
}

fn is_slime_chunk(seed: i64, chunk_x: i32, chunk_z: i32) -> bool {
    let chunk_x_sq = chunk_x.wrapping_mul(chunk_x);
    let chunk_z_sq = chunk_z.wrapping_mul(chunk_z);
    let term_x_sq = i64::from(chunk_x_sq.wrapping_mul(4_987_142));
    let term_x = i64::from(chunk_x.wrapping_mul(5_947_611));
    let term_z_sq = i64::from(chunk_z_sq).wrapping_mul(4_392_871_i64);
    let term_z = i64::from(chunk_z.wrapping_mul(389_711));
    let mut rng =
        JavaRng::new(i64::from(seed) + term_x_sq + term_x + term_z_sq + term_z ^ 987_234_911_i64);
    rng.next_i32(10) == 0
}

#[derive(Clone, Copy)]
struct JavaRng {
    seed: i64,
}

impl JavaRng {
    const MULTIPLIER: i64 = 0x5DEECE66D;
    const ADDEND: i64 = 0xB;
    const MASK: i64 = (1_i64 << 48) - 1;

    fn new(seed: i64) -> Self {
        Self {
            seed: (seed ^ Self::MULTIPLIER) & Self::MASK,
        }
    }

    fn next(&mut self, bits: i32) -> i32 {
        self.seed = self
            .seed
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::ADDEND)
            & Self::MASK;
        (self.seed >> (48 - bits)) as i32
    }

    fn next_i32(&mut self, bound: i32) -> i32 {
        if bound <= 0 {
            panic!("bound must be positive");
        }
        if (bound & -bound) == bound {
            return ((bound as i64 * self.next(31) as i64) >> 31) as i32;
        }
        loop {
            let bits = self.next(31);
            let val = bits % bound;
            if bits - val + (bound - 1) >= 0 {
                return val;
            }
        }
    }
}
