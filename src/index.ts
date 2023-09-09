import { spawn } from 'child_process';
import { mkdtemp, readFile, rm, writeFile } from 'fs/promises';
import { nanoid } from 'nanoid';
import { tmpdir } from 'os';
import { join } from 'path';

type Model = 'tiny.en' | 'tiny' | 'base.en' | 'base' | 'small.en' | 'small' | 'medium.en' | 'medium' | 'large-v1' | 'large-v2' | 'large';
type OutputFormat = 'txt' | 'vtt' | 'srt' | 'tsv' | 'json' | 'all';
type Task = 'transcribe' | 'translate';
type Language = 'af' | 'am' | 'ar' | 'as' | 'az' | 'ba' | 'be' | 'bg' | 'bn' | 'bo' | 'br' | 'bs' | 'ca' | 'cs' | 'cy' | 'da' | 'de' | 'el' | 'en' | 'es' | 'et' | 'eu' | 'fa' | 'fi' | 'fo' | 'fr' | 'gl' | 'gu' | 'ha' | 'haw' | 'he' | 'hi' | 'hr' | 'ht' | 'hu' | 'hy' | 'id' | 'is' | 'it' | 'ja' | 'jw' | 'ka' | 'kk' | 'km' | 'kn' | 'ko' | 'la' | 'lb' | 'ln' | 'lo' | 'lt' | 'lv' | 'mg' | 'mi' | 'mk' | 'ml' | 'mn' | 'mr' | 'ms' | 'mt' | 'my' | 'ne' | 'nl' | 'nn' | 'no' | 'oc' | 'pa' | 'pl' | 'ps' | 'pt' | 'ro' | 'ru' | 'sa' | 'sd' | 'si' | 'sk' | 'sl' | 'sn' | 'so' | 'sq' | 'sr' | 'su' | 'sv' | 'sw' | 'ta' | 'te' | 'tg' | 'th' | 'tk' | 'tl' | 'tr' | 'tt' | 'uk' | 'ur' | 'uz' | 'vi' | 'yi' | 'yo' | 'zh' | 'Afrikaans' | 'Albanian' | 'Amharic' | 'Arabic' | 'Armenian' | 'Assamese' | 'Azerbaijani' | 'Bashkir' | 'Basque' | 'Belarusian' | 'Bengali' | 'Bosnian' | 'Breton' | 'Bulgarian' | 'Burmese' | 'Castilian' | 'Catalan' | 'Chinese' | 'Croatian' | 'Czech' | 'Danish' | 'Dutch' | 'English' | 'Estonian' | 'Faroese' | 'Finnish' | 'Flemish' | 'French' | 'Galician' | 'Georgian' | 'German' | 'Greek' | 'Gujarati' | 'Haitian' | 'Haitian Creole' | 'Hausa' | 'Hawaiian' | 'Hebrew' | 'Hindi' | 'Hungarian' | 'Icelandic' | 'Indonesian' | 'Italian' | 'Japanese' | 'Javanese' | 'Kannada' | 'Kazakh' | 'Khmer' | 'Korean' | 'Lao' | 'Latin' | 'Latvian' | 'Letzeburgesch' | 'Lingala' | 'Lithuanian' | 'Luxembourgish' | 'Macedonian' | 'Malagasy' | 'Malay' | 'Malayalam' | 'Maltese' | 'Maori' | 'Marathi' | 'Moldavian' | 'Moldovan' | 'Mongolian' | 'Myanmar' | 'Nepali' | 'Norwegian' | 'Nynorsk' | 'Occitan' | 'Panjabi' | 'Pashto' | 'Persian' | 'Polish' | 'Portuguese' | 'Punjabi' | 'Pushto' | 'Romanian' | 'Russian' | 'Sanskrit' | 'Serbian' | 'Shona' | 'Sindhi' | 'Sinhala' | 'Sinhalese' | 'Slovak' | 'Slovenian' | 'Somali' | 'Spanish' | 'Sundanese' | 'Swahili' | 'Swedish' | 'Tagalog' | 'Tajik' | 'Tamil' | 'Tatar' | 'Telugu' | 'Thai' | 'Tibetan' | 'Turkish' | 'Turkmen' | 'Ukrainian' | 'Urdu' | 'Uzbek' | 'Valencian' | 'Vietnamese' | 'Welsh' | 'Yiddish' | 'Yoruba';

export interface WhisperOptions {
    model?: Model,
    model_dir?: string;
    device?: string;
    output_dir?: string;
    output_format?: OutputFormat;
    verbose?: boolean;
    task?: Task;
    language?: Language;
    temperature?: number;
    best_of?: number;
    beam_size?: number;
    patience?: number;
    length_penalty?: number;
    suppress_tokens?: number[];
    initial_prompt?: string;
    condition_on_previous_text?: boolean;
    fp16?: boolean;
    temperature_increment_on_fallback?: number;
    compression_ratio_threshold?: number;
    logprob_threshold?: number;
    no_speech_threshold?: number;
    word_timestamps?: boolean;
    prepend_punctuations?: string;
    append_punctuations?: string;
    threads?: number;
    silent?: boolean;
}

export interface WhisperResult {
    text: string;
    segments: WhisperResultSegment[];
    language: Language;
}

export interface WhisperResultSegment {
    id: number;
    seek: number;
    start: number;
    end: number;
    text: string;
    tokens: number[];
    temperature: number;
    avg_logprob: number;
    compression_ratio: number;
    no_speech_prob: number;
}

namespace Whisper {
    export const nativeRun = async (file: string, options: WhisperOptions, ...additionalFiles: string[]) => 
        new Promise<void>(resolve => {
            const { silent, ...procOptions } = options;
            const files = [file].concat(additionalFiles);
            const proc = spawn('whisper', Object.entries(procOptions).flatMap(([key, value]) => {
                const formattedKey = `--${key}`;
                if (Array.isArray(value)) {
                    return [formattedKey, value.join(',')];
                } else if (typeof value === 'boolean') {
                    const strBool = `${value}`;
                    const pyBool = strBool.slice(0).toUpperCase() + strBool.slice(1);
                    return [formattedKey, pyBool];
                }
                return [formattedKey, `${value}`];
            }).concat(files));
            proc.stdout.on('data', data => {
                if (silent) return;
                if (Buffer.isBuffer(data)) {
                    console.log(data.toString('utf-8'));
                } else {
                    console.log(data);
                }
            });
            proc.stdout.on('end', resolve);
            proc.stderr.on('data', data => {
                if (silent) return;
                if (Buffer.isBuffer(data)) {
                    console.error(data.toString('utf-8'));
                } else {
                    console.error(data);
                }
            });
        });

    export async function run(file: Buffer, options: WhisperOptions): Promise<WhisperResult>;
    export async function run(file: Buffer, options: WhisperOptions, ...additionalFiles: Buffer[]): Promise<WhisperResult[]>;
    export async function run(file: Buffer, options: WhisperOptions, ...additionalFiles: Buffer[]) {
        return new Promise<WhisperResult | WhisperResult[]>(async resolve => {
            const tempDir = await mkdtemp(join(tmpdir(), 'whisper-'));
            const files = [file].concat(additionalFiles);
            const filePaths = await Promise.all(
                files.map(async file => {
                    const tempFilePath = join(tempDir, nanoid());
                    await writeFile(tempFilePath, file);
                    return tempFilePath;
                })
            );
            options.output_dir = tempDir;
            options.output_format = 'json';
            options.silent = true;
            await nativeRun(filePaths[0], options, ...filePaths.slice(1));
            const results: WhisperResult[] = await Promise.all(
                filePaths.map(async filePath => 
                    readFile(filePath + '.json', 'utf-8')
                        .then(JSON.parse)
                )
            );
            await rm(tempDir, { recursive: true });
            
            resolve(results.length === 1 ? results[0] : results);
        });
    }

}

export default Whisper;