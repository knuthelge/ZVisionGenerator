<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { Alert } from '$lib/components/molecules';

  interface Props {
    /** The answer is being saved. */
    saving: boolean;
    error: string | null;
    onshow: () => void;
    onnotnow: () => void;
  }

  let { saving, error, onshow, onnotnow }: Props = $props();

  const ROWS = [
    { what: 'Checked by', curated: 'Someone looked at each one', hf: 'Nobody' },
    { what: 'Licence', curated: 'Looked at', hf: 'Varies; some have none' },
    { what: 'Previews', curated: 'Looked at', hf: 'Many have none' },
  ];
</script>

<div class="disc-optin">
  <h3>Curated is checked. Hugging Face is not.</h3>
  <p>Hugging Face tab has every LoRA the community authors list for this model. Show those here too?</p>
  <Alert tone="warning"><strong>On Hugging Face, content filtering is left to the community, so adult content will show up.</strong> Only repos their authors actually tag as adult can be, and are, excluded.</Alert>
  <table class="disc-optin-table">
    <thead>
      <tr>
        <th scope="col"><span class="sr-only">What is compared</span></th>
        <th scope="col">Curated</th>
        <th scope="col" class="hf">Hugging Face</th>
      </tr>
    </thead>
    <tbody>
      {#each ROWS as row (row.what)}
        <tr>
          <th scope="row">{row.what}</th>
          <td><span class="ok"><Icon name="check" size={14} /></span>{row.curated}</td>
          <td class="hf"><span class="warn"><Icon name="alert" size={14} /></span>{row.hf}</td>
        </tr>
      {/each}
    </tbody>
  </table>
  <p class="disc-optin-either"><span class="ok"><Icon name="check" size={14} /></span><span>Either way, ziv checks which model each file works with before it downloads, and never sends your Hugging Face login.</span></p>
  <dl class="disc-optin-keys">
    <dt>Show</dt>
    <dd>Shows Hugging Face results in Discover. It's saved in ziv on this computer, so it also applies when you open ziv from your phone or another device. Turn it off any time under the search box.</dd>
    <dt>Not now</dt>
    <dd>Back to Curated. Nothing is saved, and you're asked again next time.</dd>
  </dl>
  {#if error}<Alert tone="error" live>{error}</Alert>{/if}
  <div class="disc-optin-foot">
    <button type="button" class="ui-btn" disabled={saving} onclick={onnotnow}>Not now</button>
    <button type="button" class="ui-btn ui-btn-primary" disabled={saving} aria-busy={saving || undefined} onclick={onshow}>Show Hugging Face results</button>
  </div>
</div>
